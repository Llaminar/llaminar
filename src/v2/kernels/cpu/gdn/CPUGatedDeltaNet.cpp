/**
 * @file CPUGatedDeltaNet.cpp
 * @brief CPU implementation of delta rule recurrence for GDN linear attention
 *
 * CPU state owns a fixed eight-position checkpoint, phase, and raw update log.
 * Serial and grouped queries use identical backwards contractions and increasing-
 * key FP32 FMAs. Grouped queries share checkpoint loads; rounded prefix updates
 * construct each required checkpoint once. Acceptance and snapshots preserve
 * pending history and never flush a partial epoch. Independent heads own the
 * same program for decode, prefill, merged-QKV and ragged request batches.
 * Per-head completed observations are integer-reduced by that same team and
 * published once per call, retaining both original counts and numeric sums.
 *
 * The kernel owns ALL preprocessing:
 * - Q/K normalization by sqrt(sum(x*x) + 1e-6) when use_qk_l2norm is true
 * - Query scaling by 1/sqrt(d_k)
 * - Gate computation: g = -exp(A_log) * softplus(alpha + dt_bias)
 * - Beta sigmoid: beta_sig = sigmoid(beta_raw)
 *
 * Reference: torch_recurrent_gated_delta_rule() and torch_chunk_gated_delta_rule()
 *            from HuggingFace transformers 5.4.0
 */

#include "CPUGatedDeltaNet.h"
#include "CPUFixedEpochGDN.h"
#include "../../../utils/CPUFeatures.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include <vector>

#include "../simd/AVX2Helpers.h"

namespace llaminar2
{
    CPUGatedDeltaNet::CPUGatedDeltaNet()
        : epoch_counter_identities_{
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_epoch_projection_groups", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_epoch_grouped_projection_rows", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_epoch_checkpoint_flushes", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_candidate_rows", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_checkpoint_copies", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_checkpoint_copy_bytes", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_initial_checkpoint_fused_stores", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_initial_checkpoint_fused_store_bytes", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_derived_checkpoint_writes", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_derived_checkpoint_write_bytes", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_explicit_checkpoint_copy_passes", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_explicit_checkpoint_copy_bytes", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_raw_log_records", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_raw_log_copy_bytes", "recurrence", "cpu"),
            PerfStatsCollector::bindCounter("kernel", "cpu_gdn_shared_mutable_bank_copy_bytes", "recurrence", "cpu")}
    {
    }

    bool CPUGatedDeltaNet::beginEpochCounterObservations()
    {
        // A concurrently published collector generation may differ between
        // arriving workers. Only this existing team owner selects the branch;
        // its implicit barrier makes one choice visible before any workshare.
        OMP_SINGLE([&]() {
            epoch_counter_collection_enabled_ = PerfStatsCollector::isDomainEnabled("kernel");
            if (epoch_counter_collection_enabled_) epoch_counter_observations_ = {};
        });
        return epoch_counter_collection_enabled_;
    }

    void CPUGatedDeltaNet::publishEpochCounterObservations(int dk, int dv) const
    {
        const auto &o = epoch_counter_observations_;
        if (o.heads == 0) return;
        const CPUGDNStateLayout layout(1, dk, dv);
        const double matrix_bytes = static_cast<double>(layout.matrixFloats()) * sizeof(float);
        const auto emit = [&](size_t index, std::uint64_t count, double value)
        {
            PerfStatsCollector::addCounter(epoch_counter_identities_[index],
                PerfStatsCollector::CounterObservationBatch{.observations=count, .value=value});
        };
        emit(0, o.heads, static_cast<double>(o.projection_groups));
        emit(1, o.heads, static_cast<double>(o.grouped_rows));
        emit(2, o.heads, static_cast<double>(o.checkpoint_flushes));
        if (o.shared_heads == 0) return;
        emit(3, o.shared_heads, static_cast<double>(o.candidate_rows));
        emit(4, o.shared_heads, static_cast<double>(o.shared_heads));
        emit(5, o.shared_heads, o.shared_heads * matrix_bytes);
        emit(6, o.shared_heads, static_cast<double>(o.shared_heads));
        emit(7, o.shared_heads, o.shared_heads * matrix_bytes);
        emit(8, o.shared_heads, static_cast<double>(o.shared_flushes));
        emit(9, o.shared_heads, o.shared_flushes * matrix_bytes);
        emit(10, o.shared_heads, 0.0);
        emit(11, o.shared_heads, 0.0);
        emit(12, o.shared_heads, static_cast<double>(o.raw_records));
        emit(13, o.shared_heads, static_cast<double>(o.raw_records) * layout.recordFloats() * sizeof(float));
        emit(14, o.shared_heads, 0.0);
    }

    /**
     * @brief Test two byte ranges for overlap and fail closed on address overflow.
     */
    static bool byteRangesOverlap(
        const void *left,
        size_t left_bytes,
        const void *right,
        size_t right_bytes)
    {
        const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
        const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
        if (left_bytes > std::numeric_limits<uintptr_t>::max() - left_begin ||
            right_bytes > std::numeric_limits<uintptr_t>::max() - right_begin)
        {
            return true;
        }

        const uintptr_t left_end = left_begin + left_bytes;
        const uintptr_t right_end = right_begin + right_bytes;
        return left_begin < right_end && right_begin < left_end;
    }

    bool CPUGatedDeltaNet::resetGPUState(void *stream)
    {
        (void)stream;
        shared_valid_=false;
        request_state_bank_.clear();
        request_state_size_ = 0;
        request_state_capacity_ = 0;
        return true;
    }

    bool CPUGatedDeltaNet::ensureRequestStateBank(
        int request_count,
        int state_floats,
        const float *request_zero_state)
    {
        if (request_count <= 0 || state_floats <= 0 || !request_zero_state)
            return false;

        if (request_state_size_ != state_floats)
        {
            request_state_bank_.assign(
                static_cast<size_t>(request_count) * state_floats,
                0.0f);
            request_state_size_ = state_floats;
            request_state_capacity_ = request_count;
        }
        else if (request_state_capacity_ < request_count)
        {
            request_state_bank_.resize(
                static_cast<size_t>(request_count) * state_floats,
                0.0f);
            request_state_capacity_ = request_count;
        }

        /* Request zero remains the public host prefix-state representation. */
        std::memcpy(
            request_state_bank_.data(),
            request_zero_state,
            static_cast<size_t>(state_floats) * sizeof(float));
        return true;
    }

    void CPUGatedDeltaNet::bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size)
    {
        shared_valid_=false;
        verifier_state_capture_ = workspace;
        verifier_state_capture_rows_ = rows;
        verifier_state_capture_size_ = state_size;
    }

    /**
     * @brief Borrow compact arena buffers and retire local publication eligibility.
     * @param slots Immutable checkpoint/raw-log payload, or null when clearing.
     * @param slot_floats Logical admitted payload extent in FP32-sized elements.
     * @param rows Fixed admitted flattened candidate capacity.
     * @throws std::invalid_argument for partial bindings or byte/prefix overflow.
     * A stage may own a completed capture when another graph rebinds this shared
     * backend; this operation therefore must not retire stage-owned slot headers.
     */
    void CPUGatedDeltaNet::bindCPUSharedStateCapture(float *slots,size_t slot_floats,int rows)
    {
        shared_valid_=false;
        if ((slots && (!slot_floats || rows<=0 || rows>std::numeric_limits<int>::max()-8)) ||
            (!slots && (slot_floats || rows)) || slot_floats>std::numeric_limits<size_t>::max()/sizeof(float))
            throw std::invalid_argument("CPU shared epoch capture requires complete admitted slot binding");
        shared_slots_=slots; shared_slot_floats_=slot_floats; shared_rows_=rows;
    }

    /**
     * @brief Authenticate logical capacity and geometry before recurrence writes.
     * @param requests Fixed number of graph request partitions.
     * @param width Live input-row width per request.
     * @param heads Local value-head count.
     * @param dk Key width of each canonical head.
     * @param dv Value width of each canonical head.
     * @return True for a complete immutable-slot binding with admitted capacity.
     * @throws std::invalid_argument for unsupported canonical state geometry.
     * @throws std::overflow_error if the canonical bank exceeds its INT ABI.
     * Physical spans retain admitted width; accepted indices retain live width.
     */
    bool CPUGatedDeltaNet::prepareShared(int requests,int width,int heads,int dk,int dv)
    {
        shared_valid_=false;
        const CPUGDNStateLayout l(heads,dk,dv);
        if (requests<=0 || shared_rows_<=0 || shared_rows_%requests || width>shared_rows_/requests) return false;
        shared_live_width_=width;
        width=shared_rows_/requests;
        if (!shared_slots_ || width<=0 || requests<=0 ||
            static_cast<size_t>(requests)*width>static_cast<size_t>(shared_rows_) ||
            CPUSharedGDNSnapshots::headFloats(l,width)>shared_slot_floats_/static_cast<size_t>(heads)/requests) return false;
        shared_heads_=heads; shared_dk_=dk; shared_dv_=dv;
        shared_width_=width; shared_requests_=requests;
        return true;
    }

    /**
     * @brief Check every head before publishing a locally bound candidate.
     * @param row Flattened candidate index using the current live request width.
     * @return True only for valid local lifecycle and complete candidate headers.
     * Invalid candidates leave the immutable payload and canonical state intact.
     */
    bool CPUGatedDeltaNet::validSharedRow(int row) const
    {
        if (!shared_valid_ || row<0 || row>=shared_requests_*shared_live_width_) return false;
        const CPUGDNStateLayout l(shared_heads_,shared_dk_,shared_dv_);
        const int request=row/shared_live_width_,local=row%shared_live_width_;
        const size_t span=CPUSharedGDNSnapshots::headFloats(l,shared_width_);
        for(int head=0;head<shared_heads_;++head)
            if (!CPUSharedGDNSnapshots(l,shared_slots_+(static_cast<size_t>(request)*shared_heads_+head)*span,shared_width_).validRow(local)) return false;
        return true;
    }

    /**
     * @brief Serialize one complete locally bound candidate without arithmetic.
     * @param destination Writable canonical bank, disjoint from retained slots.
     * @param row Flattened candidate index using the live request width.
     * @return True after every head is copied, or false before mutation on invalid
     *         lifecycle, candidate metadata, destination, or buffer overlap.
     */
    bool CPUGatedDeltaNet::materializeShared(float *destination,int row)
    {
        if (!destination || !validSharedRow(row)) return false;
        const size_t bytes=static_cast<size_t>(CPUGDNStateLayout(shared_heads_,shared_dk_,shared_dv_).floats())*sizeof(float);
        if (byteRangesOverlap(destination,bytes,shared_slots_,shared_slot_floats_*sizeof(float))) return false;
        const CPUGDNStateLayout l(shared_heads_,shared_dk_,shared_dv_);
        const int request=row/shared_live_width_,local=row%shared_live_width_;
        const size_t span=CPUSharedGDNSnapshots::headFloats(l,shared_width_);
        for(int head=0;head<shared_heads_;++head)
            CPUSharedGDNSnapshots(l,shared_slots_+(static_cast<size_t>(request)*shared_heads_+head)*span,shared_width_)
                .materialize(local,destination+static_cast<size_t>(head)*l.headFloats());
        return true;
    }

    /**
     * @copydoc ITensorGatedDeltaNet::restoreCPUSharedStateRows
     *
     * Validate the stage-owned generation and all selected head prefixes before
     * modifying the canonical request bank. Rejection preserves that request's
     * live bank. Explicit transactions borrow their stage's lifetime rather than
     * this backend's current binding, so unrelated kernel clears remain harmless.
     * No allocation, row replay, or partial checkpoint advancement occurs here.
     */
    bool CPUGatedDeltaNet::restoreCPUSharedStateRows(const CPUSharedGDNTransaction &t,
        float *destination,const int *rows)
    {
        // The owning stage authenticates lifetime independently of the shared
        // kernel binding. Another graph role may clear/rebind this backend while
        // a completed stage-owned transaction is still awaiting publication.
        if (!t.slots || !destination || !rows || t.requests<=0 || t.width<=0 || t.live_width<=0 || t.live_width>t.width ||
            !t.generation_authority || *t.generation_authority!=t.generation) return false;
        const CPUGDNStateLayout l(t.heads,t.dk,t.dv);
        const size_t span=CPUSharedGDNSnapshots::headFloats(l,t.width);
        if (span>t.slot_floats/static_cast<size_t>(t.heads)/t.requests ||
            (t.requests>1 && (t.requests>request_state_capacity_ || request_state_size_!=l.floats())) ||
            t.slot_floats>std::numeric_limits<size_t>::max()/sizeof(float) ||
            byteRangesOverlap(destination,static_cast<size_t>(l.floats())*sizeof(float),t.slots,t.slot_floats*sizeof(float))) return false;
        for(int request=0;request<t.requests;++request) {
            if (rows[request]<0) continue;
            if (rows[request]/t.live_width!=request) return false;
            const int row=rows[request]%t.live_width;
            for(int head=0;head<t.heads;++head)
                if (!CPUSharedGDNSnapshots(l,t.slots+(static_cast<size_t>(request)*t.heads+head)*span,t.width).validRow(row)) return false;
        }
        for(int request=0;request<t.requests;++request) {
            if (rows[request]<0) continue;
            float *target=t.requests==1 ? destination : request_state_bank_.data()+static_cast<size_t>(request)*l.floats();
            for(int head=0;head<t.heads;++head)
                CPUSharedGDNSnapshots(l,t.slots+(static_cast<size_t>(request)*t.heads+head)*span,t.width)
                    .materialize(rows[request]%t.live_width,target+static_cast<size_t>(head)*l.headFloats());
            if (request==0 && t.requests>1) std::memcpy(destination,target,static_cast<size_t>(l.floats())*sizeof(float));
            PerfStatsCollector::addCounter("kernel","cpu_gdn_shared_accepted_materializations",1,"publication","cpu");
            PerfStatsCollector::addCounter("kernel","cpu_gdn_shared_accepted_materialization_bytes",
                static_cast<double>(l.floats())*sizeof(float),"publication","cpu");
        }
        return true;
    }

    bool CPUGatedDeltaNet::restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream)
    {
        if (shared_slots_) return materializeShared(dst_state,row);
        if (row < 0 || row >= verifier_state_capture_rows_)
            return false;
        return restoreStateFromSnapshot(
            dst_state,
            verifier_state_capture_,
            row,
            verifier_state_capture_size_,
            verifier_state_capture_size_,
            stream);
    }

    bool CPUGatedDeltaNet::restoreVerifierStateCaptureRows(
        float *dst_state,
        const int *host_row_indices,
        int request_count,
        void *stream)
    {
        (void)stream;
        if (!dst_state || !host_row_indices || request_count <= 0 ||
            !verifier_state_capture_ || verifier_state_capture_size_ <= 0)
        {
            return false;
        }

        /*
         * Publication is one logical transaction across the request bank.
         * Reject every invalid row before copying any slot so callers never
         * inherit a partially committed speculative timeline.
         */
        for (int request = 0; request < request_count; ++request)
        {
            const int row = host_row_indices[request];
            if (row >= verifier_state_capture_rows_ ||
                (shared_slots_ && row>=0 && (!validSharedRow(row) || row/shared_live_width_!=request)))
                return false;
        }

        if (request_count == 1)
        {
            /*
             * A one-request transaction still arrives through the grouped row
             * vector API, but its live recurrence state is the public state
             * tensor rather than an internal request bank.  Commit the selected
             * post-row snapshot directly.  This is a native byte copy only; it
             * never replays a recurrent row or changes arithmetic ordering.
             */
            const int row = host_row_indices[0];
            if (row >= 0)
            {
                if (shared_slots_) { if (!materializeShared(dst_state,row)) return false; }
                else std::memcpy(
                    dst_state,
                    verifier_state_capture_ +
                        static_cast<size_t>(row) * verifier_state_capture_size_,
                    static_cast<size_t>(verifier_state_capture_size_) * sizeof(float));
            }

            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_gdn_request_batched_state_publications",
                1.0,
                "decode",
                "cpu",
                {{"request_count", "1"},
                 {"publication_policy", "single_request_snapshot_copy"}});
            return true;
        }

        if (request_count > request_state_capacity_ ||
            request_state_size_ <= 0 ||
            verifier_state_capture_size_ < request_state_size_)
        {
            return false;
        }

        for (int request = 0; request < request_count; ++request)
        {
            const int row = host_row_indices[request];
            if (row < 0)
                continue;
            float *destination =
                request_state_bank_.data() +
                static_cast<size_t>(request) * request_state_size_;
            const float *source = shared_slots_ ? nullptr :
                verifier_state_capture_ + static_cast<size_t>(row) * verifier_state_capture_size_;
            if (shared_slots_) { if (!materializeShared(destination,row)) return false; }
            else std::memcpy(
                destination,
                source,
                static_cast<size_t>(request_state_size_) * sizeof(float));
            if (request == 0)
            {
                std::memcpy(
                    dst_state,
                    destination,
                    static_cast<size_t>(request_state_size_) * sizeof(float));
            }
        }

        PerfStatsCollector::addCounter(
            "kernel",
            "cpu_gdn_request_batched_state_publications",
            1.0,
            "decode",
            "cpu",
            {{"request_count", std::to_string(request_count)},
             {"publication_policy", "request_bank_snapshot_copy"}});
        return true;
    }

    // =========================================================================
    // Scratch buffer management (grow-only, eliminates per-call allocations)
    // =========================================================================

    void CPUGatedDeltaNet::ensureScratch(int seq_len, int n_heads, int d_k, int /*d_v*/)
    {
        const size_t qk_total = static_cast<size_t>(seq_len) * n_heads * d_k;
        const size_t gate_total = static_cast<size_t>(seq_len) * n_heads;

        if (q_scratch_.size() < qk_total)
            q_scratch_.resize(qk_total);
        if (k_scratch_.size() < qk_total)
            k_scratch_.resize(qk_total);
        if (gate_scratch_.size() < gate_total)
            gate_scratch_.resize(gate_total);
        if (beta_sig_scratch_.size() < gate_total)
            beta_sig_scratch_.resize(gate_total);
    }

    // =========================================================================
    // Preprocessing helpers
    // =========================================================================

    void CPUGatedDeltaNet::computeGates(
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *g_out, float *beta_sig_out,
        int seq_len, int n_heads)
    {
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                const int idx = t * n_heads + h;

                const float x = alpha[idx] + dt_bias[h];
                const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
                // GGUF stores -exp(A_log), so use it directly as the decay coefficient.
                // g = -exp(A_log) * softplus(alpha + dt_bias) = stored_value * softplus(...)
                g_out[idx] = A_log[h] * sp;

                beta_sig_out[idx] = 1.0f / (1.0f + std::exp(-beta_raw[idx]));
            }
        }
    }

    // Named ISA implementations: l2normalize_vec (single vector)
    static void l2normalize_vec_scalar(float *vec, int head_dim, float eps)
    {
        float norm_sq = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            norm_sq += vec[d] * vec[d];
        const float inv_norm = 1.0f / std::sqrt(norm_sq + eps);
        for (int d = 0; d < head_dim; ++d)
            vec[d] *= inv_norm;
    }

#if defined(__AVX2__)
    static void l2normalize_vec_avx2(float *vec, int head_dim, float eps)
    {
        const int hd_vec = head_dim & ~7;
        __m256 vsum = _mm256_setzero_ps();
        int d = 0;
        for (; d < hd_vec; d += 8)
        {
            __m256 vv = _mm256_loadu_ps(vec + d);
            vsum = _mm256_fmadd_ps(vv, vv, vsum);
        }
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float norm_sq = _mm_cvtss_f32(lo);
        for (; d < head_dim; ++d)
            norm_sq += vec[d] * vec[d];

        const float inv_norm = 1.0f / std::sqrt(norm_sq + eps);
        const __m256 vinv = _mm256_set1_ps(inv_norm);
        d = 0;
        for (; d < hd_vec; d += 8)
            _mm256_storeu_ps(vec + d, _mm256_mul_ps(_mm256_loadu_ps(vec + d), vinv));
        for (; d < head_dim; ++d)
            vec[d] *= inv_norm;
    }
#endif

#if defined(__AVX512F__)
    static void l2normalize_vec_avx512(float *vec, int head_dim, float eps)
    {
        const int hd_vec = head_dim & ~15;
        __m512 vsum = _mm512_setzero_ps();
        int d = 0;
        for (; d < hd_vec; d += 16)
        {
            __m512 vv = _mm512_loadu_ps(vec + d);
            vsum = _mm512_fmadd_ps(vv, vv, vsum);
        }
        float norm_sq = _mm512_reduce_add_ps(vsum);
        for (; d < head_dim; ++d)
            norm_sq += vec[d] * vec[d];

        const float inv_norm = 1.0f / std::sqrt(norm_sq + eps);
        const __m512 vinv = _mm512_set1_ps(inv_norm);
        d = 0;
        for (; d < hd_vec; d += 16)
            _mm512_storeu_ps(vec + d, _mm512_mul_ps(_mm512_loadu_ps(vec + d), vinv));
        for (; d < head_dim; ++d)
            vec[d] *= inv_norm;
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void l2normalize_vec_avx2(float *vec, int head_dim, float eps)
    {
        l2normalize_vec_scalar(vec, head_dim, eps);
    }
#endif
#if !defined(__AVX512F__)
    static void l2normalize_vec_avx512(float *vec, int head_dim, float eps)
    {
        l2normalize_vec_avx2(vec, head_dim, eps);
    }
#endif

    static inline void l2normalize_vec(float *vec, int head_dim, float eps)
    {
        ISA_DISPATCH_VOID(l2normalize_vec, vec, head_dim, eps);
    }

    void CPUGatedDeltaNet::l2normalize(float *data, int seq_len, int n_heads, int head_dim)
    {
        constexpr float eps = 1e-6f;

        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                float *vec = data + t * n_heads * head_dim + h * head_dim;
                l2normalize_vec(vec, head_dim, eps);
            }
        }
    }

    // =========================================================================
    // Named ISA implementations: QK preprocessing helpers
    // (shared between recurrent_step and chunk_forward)
    // =========================================================================

    // Qwen/FLA adds epsilon to squared norm; clamping the norm itself
    // changes low-magnitude Q/K values and the recurrent state. Preserve the
    // existing ISA reduction trees and fused query scaling association.
    // L2-normalize Q with fused scale, L2-normalize K (no scale)
    static void gdn_preprocess_qk_l2norm_scalar(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        float nq = 0.0f;
        for (int d = 0; d < dim; ++d)
            nq += q_src[d] * q_src[d];
        const float inv_q = scale / std::sqrt(nq + eps);
        for (int d = 0; d < dim; ++d)
            q_dst[d] = q_src[d] * inv_q;

        float nk = 0.0f;
        for (int d = 0; d < dim; ++d)
            nk += k_src[d] * k_src[d];
        const float inv_k = 1.0f / std::sqrt(nk + eps);
        for (int d = 0; d < dim; ++d)
            k_dst[d] = k_src[d] * inv_k;
    }

#if defined(__AVX2__)
    static void gdn_preprocess_qk_l2norm_avx2(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        avx2::l2norm_scale(q_src, q_dst, dim, scale, eps);
        avx2::l2norm_scale(k_src, k_dst, dim, 1.0f, eps);
    }
#endif

#if defined(__AVX512F__)
    static void gdn_preprocess_qk_l2norm_avx512(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        const int hd_vec = dim & ~15;
        // Q: fused L2-normalize + scale
        {
            __m512 vsum = _mm512_setzero_ps();
            int d = 0;
            for (; d < hd_vec; d += 16)
            {
                __m512 vv = _mm512_loadu_ps(q_src + d);
                vsum = _mm512_fmadd_ps(vv, vv, vsum);
            }
            float norm_sq = _mm512_reduce_add_ps(vsum);
            for (; d < dim; ++d)
                norm_sq += q_src[d] * q_src[d];
            const float inv = scale / std::sqrt(norm_sq + eps);
            const __m512 vinv = _mm512_set1_ps(inv);
            d = 0;
            for (; d < hd_vec; d += 16)
                _mm512_storeu_ps(q_dst + d,
                                 _mm512_mul_ps(_mm512_loadu_ps(q_src + d), vinv));
            for (; d < dim; ++d)
                q_dst[d] = q_src[d] * inv;
        }
        // K: L2-normalize only
        {
            __m512 vsum = _mm512_setzero_ps();
            int d = 0;
            for (; d < hd_vec; d += 16)
            {
                __m512 vv = _mm512_loadu_ps(k_src + d);
                vsum = _mm512_fmadd_ps(vv, vv, vsum);
            }
            float norm_sq = _mm512_reduce_add_ps(vsum);
            for (; d < dim; ++d)
                norm_sq += k_src[d] * k_src[d];
            const float inv = 1.0f / std::sqrt(norm_sq + eps);
            const __m512 vinv = _mm512_set1_ps(inv);
            d = 0;
            for (; d < hd_vec; d += 16)
                _mm512_storeu_ps(k_dst + d,
                                 _mm512_mul_ps(_mm512_loadu_ps(k_src + d), vinv));
            for (; d < dim; ++d)
                k_dst[d] = k_src[d] * inv;
        }
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void gdn_preprocess_qk_l2norm_avx2(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        gdn_preprocess_qk_l2norm_scalar(q_src, k_src, q_dst, k_dst, dim, scale, eps);
    }
#endif
#if !defined(__AVX512F__)
    static void gdn_preprocess_qk_l2norm_avx512(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        gdn_preprocess_qk_l2norm_avx2(q_src, k_src, q_dst, k_dst, dim, scale, eps);
    }
#endif

    static inline void gdn_preprocess_qk_l2norm(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        ISA_DISPATCH_VOID(gdn_preprocess_qk_l2norm, q_src, k_src, q_dst, k_dst, dim, scale, eps);
    }

    // Scale Q, copy K unchanged
    static void gdn_preprocess_qk_scale_scalar(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale)
    {
        for (int d = 0; d < dim; ++d)
            q_dst[d] = q_src[d] * scale;
        std::memcpy(k_dst, k_src, dim * sizeof(float));
    }

#if defined(__AVX2__)
    static void gdn_preprocess_qk_scale_avx2(
        const float *q_src, const float * /*k_src*/,
        float *q_dst, float *k_dst,
        int dim, float scale)
    {
        avx2::copy_scale(q_dst, q_src, scale, dim);
        // k_dst filled by caller via memcpy (passed through)
        (void)k_dst;
    }
#endif

#if defined(__AVX512F__)
    static void gdn_preprocess_qk_scale_avx512(
        const float *q_src, const float * /*k_src*/,
        float *q_dst, float * /*k_dst*/,
        int dim, float scale)
    {
        const int hd_vec = dim & ~15;
        const __m512 vscale = _mm512_set1_ps(scale);
        int d = 0;
        for (; d < hd_vec; d += 16)
            _mm512_storeu_ps(q_dst + d,
                             _mm512_mul_ps(_mm512_loadu_ps(q_src + d), vscale));
        for (; d < dim; ++d)
            q_dst[d] = q_src[d] * scale;
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void gdn_preprocess_qk_scale_avx2(
        const float *q_src, const float * /*k_src*/,
        float *q_dst, float *k_dst,
        int dim, float scale)
    {
        (void)k_dst;
        gdn_preprocess_qk_scale_scalar(q_src, nullptr, q_dst, k_dst, dim, scale);
    }
#endif
#if !defined(__AVX512F__)
    static void gdn_preprocess_qk_scale_avx512(
        const float *q_src, const float * /*k_src*/,
        float *q_dst, float * /*k_dst*/,
        int dim, float scale)
    {
        gdn_preprocess_qk_scale_avx2(q_src, nullptr, q_dst, nullptr, dim, scale);
    }
#endif

    static inline void gdn_preprocess_qk_scale(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale)
    {
        switch (activeISALevel())
        {
        case ISALevel::AVX512:
            gdn_preprocess_qk_scale_avx512(q_src, k_src, q_dst, k_dst, dim, scale);
            std::memcpy(k_dst, k_src, dim * sizeof(float));
            break;
        case ISALevel::AVX2:
            gdn_preprocess_qk_scale_avx2(q_src, k_src, q_dst, k_dst, dim, scale);
            std::memcpy(k_dst, k_src, dim * sizeof(float));
            break;
        default:
            gdn_preprocess_qk_scale_scalar(q_src, k_src, q_dst, k_dst, dim, scale);
            break;
        }
    }

    bool CPUGatedDeltaNet::recurrent_step(
        const float *q, const float *k, const float *v,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int n_heads, int d_k, int d_v, bool use_qk_l2norm)
    {
        shared_valid_=false;
        if (!q || !k || !v || !alpha || !beta_raw || !A_log || !dt_bias || !output || !state ||
            n_heads <= 0 || d_k <= 0 || d_v <= 0 || d_k > 512 || d_v > 512)
            return false;
        const CPUGDNStateLayout layout(n_heads, d_k, d_v);
        if (!layout.valid(state)) return false;
        const bool capture = verifier_state_capture_ != nullptr;
        if (capture && (!shared_slots_ || verifier_state_capture_!=shared_slots_))
            throw std::invalid_argument("CPU GDN production verifier requires bound shared epoch slots; dense snapshots require explicit serialization API");
        if (capture && (verifier_state_capture_rows_ < 1 || verifier_state_capture_size_ < layout.floats() ||
            byteRangesOverlap(state, static_cast<size_t>(layout.floats()) * sizeof(float),
                verifier_state_capture_, shared_slot_floats_ * sizeof(float))))
            return false;
        if (capture && !prepareShared(1,1,n_heads,d_k,d_v)) return false;
        PerfStatsCollector::addCounter("kernel", "cpu_gdn_epoch_readout_rows",
            static_cast<double>(n_heads), "recurrence", "cpu");
        const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        const bool observe_epoch = beginEpochCounterObservations();
        auto &observations = epoch_counter_observations_;
        // Each member keeps its returned local choice through publication;
        // subsequent collector reloads cannot change workshare control flow.
        const auto advance_head = [&](int h) -> CPUFixedEpochObservations
        {
                alignas(64) float query[512], key[512];
                if (use_qk_l2norm)
                    gdn_preprocess_qk_l2norm(q + h*d_k, k + h*d_k, query, key, d_k, scale, 1e-6f);
                else gdn_preprocess_qk_scale(q + h*d_k, k + h*d_k, query, key, d_k, scale);
                const float x = alpha[h] + dt_bias[h];
                const float decay = std::exp(A_log[h] * (x > 20.0f ? x : std::log1p(std::exp(x))));
                const float beta = 1.0f / (1.0f + std::exp(-beta_raw[h]));
                float *initial = state + static_cast<size_t>(h) * layout.headFloats();
                if (capture)
                {
                    CPUSharedGDNSnapshots shared(layout,shared_slots_+static_cast<size_t>(h)*CPUSharedGDNSnapshots::headFloats(layout,shared_width_),shared_width_);
                    return cpu_gdn_epoch::runShared(layout,initial,query,key,v+h*d_v,&decay,&beta,
                        1,d_k,d_v,1,output+h*d_v,d_v,shared);
                }
                else
                    return cpu_gdn_epoch::run(layout,initial,initial,query,key,v+h*d_v,
                        &decay,&beta,1,d_k,d_v,1,output+h*d_v,d_v);
        };
        auto work = [&]()
        {
            if (observe_epoch)
            {
#pragma omp for schedule(static) reduction(cpu_gdn_epoch_observations : observations)
                for (int h = 0; h < n_heads; ++h)
                    observations += advance_head(h);
            }
            else
            {
#pragma omp for schedule(static)
                for (int h = 0; h < n_heads; ++h)
                    (void)advance_head(h);
            }
        };
        OMP_WORKSHARE_REGION(work);
        if (observe_epoch) OMP_SINGLE([&]() { publishEpochCounterObservations(d_k, d_v); });
        if (capture) shared_valid_=true;
        return true;
    }

    bool CPUGatedDeltaNet::chunk_forward(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        int chunk_size, bool use_qk_l2norm)
    {
        shared_valid_=false;
        if (n_heads <= 0 || d_k <= 0 || d_v <= 0 || d_k > 512 || d_v > 512)
            return false;
        const int state_floats = CPUGDNStateLayout(n_heads, d_k, d_v).floats();
        if (verifier_state_capture_ && (!shared_slots_ || verifier_state_capture_!=shared_slots_))
            throw std::invalid_argument("CPU GDN production verifier requires bound shared epoch slots; dense snapshots require explicit serialization API");
        if (verifier_state_capture_ &&
            (verifier_state_capture_rows_ < seq_len || verifier_state_capture_size_ < state_floats))
            return false;
        float *state_snapshots = nullptr;
        int snapshot_stride_floats = 0;
        int max_snapshot_rows = 0;
        if (verifier_state_capture_ &&
            verifier_state_capture_rows_ > 0 &&
            verifier_state_capture_size_ >= state_floats)
        {
            state_snapshots = verifier_state_capture_;
            snapshot_stride_floats = verifier_state_capture_size_;
            max_snapshot_rows = verifier_state_capture_rows_;
        }
        const ChunkInputView input{
            .q = Q,
            .k = K,
            .v = V,
            .q_row_stride = n_heads * d_k,
            .k_row_stride = n_heads * d_k,
            .v_row_stride = n_heads * d_v,
            .n_k_heads = n_heads,
            .global_v_head_offset = 0,
            .layout_name = "separate_qkv"};
        return chunkForwardImpl(
            input, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            state_snapshots
                ? InputStateDisposition::PreserveInitialState
                : InputStateDisposition::PublishTerminalState);
    }

    bool CPUGatedDeltaNet::chunkForwardMergedQKV(
        const float *merged_qkv, int qkv_stride,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_k_heads, int n_heads, int d_k, int d_v,
        int global_v_head_offset, int chunk_size,
        bool use_qk_l2norm)
    {
        shared_valid_=false;
        if (!merged_qkv || seq_len <= 0 || n_k_heads <= 0 ||
            n_heads <= 0 || d_k <= 0 || d_v <= 0)
        {
            return false;
        }

        const int q_src_dim = n_k_heads * d_k;
        const int k_src_dim = n_k_heads * d_k;
        const int v_dim = n_heads * d_v;
        if (qkv_stride < q_src_dim + k_src_dim + v_dim)
            return false;

        if (n_heads <= 0 || d_k <= 0 || d_v <= 0 || d_k > 512 || d_v > 512)
            return false;
        const int state_floats = CPUGDNStateLayout(n_heads, d_k, d_v).floats();
        if (verifier_state_capture_ && (!shared_slots_ || verifier_state_capture_!=shared_slots_))
            throw std::invalid_argument("CPU GDN production verifier requires bound shared epoch slots; dense snapshots require explicit serialization API");
        if (verifier_state_capture_ &&
            (verifier_state_capture_rows_ < seq_len || verifier_state_capture_size_ < state_floats))
            return false;
        float *state_snapshots = nullptr;
        int snapshot_stride_floats = 0;
        int max_snapshot_rows = 0;
        if (verifier_state_capture_ &&
            verifier_state_capture_rows_ > 0 &&
            verifier_state_capture_size_ >= state_floats)
        {
            state_snapshots = verifier_state_capture_;
            snapshot_stride_floats = verifier_state_capture_size_;
            max_snapshot_rows = verifier_state_capture_rows_;
        }

        if (state_snapshots)
        {
            return chunkForwardMergedQKVWithStateSnapshots(
                merged_qkv, qkv_stride,
                alpha, beta_raw, A_log, dt_bias,
                output, state,
                seq_len, n_k_heads, n_heads, d_k, d_v,
                global_v_head_offset, chunk_size, use_qk_l2norm,
                state_snapshots, snapshot_stride_floats,
                max_snapshot_rows);
        }

        const ChunkInputView input{
            .q = merged_qkv,
            .k = merged_qkv + q_src_dim,
            .v = merged_qkv + q_src_dim + k_src_dim,
            .q_row_stride = qkv_stride,
            .k_row_stride = qkv_stride,
            .v_row_stride = qkv_stride,
            .n_k_heads = n_k_heads,
            .global_v_head_offset = global_v_head_offset,
            .layout_name = "merged_qkv"};
        return chunkForwardImpl(
            input, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows,
            InputStateDisposition::PublishTerminalState);
    }

    bool CPUGatedDeltaNet::chunkForwardWithStateSnapshots(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        int chunk_size, bool use_qk_l2norm,
        float *state_snapshots, int snapshot_stride_floats,
        int max_snapshot_rows)
    {
        const ChunkInputView input{
            .q = Q,
            .k = K,
            .v = V,
            .q_row_stride = n_heads * d_k,
            .k_row_stride = n_heads * d_k,
            .v_row_stride = n_heads * d_v,
            .n_k_heads = n_heads,
            .global_v_head_offset = 0,
            .layout_name = "separate_qkv"};
        return chunkForwardImpl(
            input, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows,
            InputStateDisposition::PublishTerminalState);
    }

    bool CPUGatedDeltaNet::restoreStateFromSnapshot(
        float *state, const float *state_snapshots,
        int snapshot_row, int snapshot_stride_floats,
        int state_floats, void *stream)
    {
        (void)stream;
        if (shared_slots_ && state_snapshots==shared_slots_) return materializeShared(state,snapshot_row);
        if (!state || !state_snapshots || snapshot_row < 0 ||
            snapshot_stride_floats < state_floats || state_floats < 0)
            return false;

        std::memcpy(state,
                    state_snapshots + static_cast<size_t>(snapshot_row) * snapshot_stride_floats,
                    static_cast<size_t>(state_floats) * sizeof(float));
        return true;
    }

    bool CPUGatedDeltaNet::chunkForwardMergedQKVWithStateSnapshots(
        const float *merged_qkv, int qkv_stride,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_k_heads, int n_heads, int d_k, int d_v,
        int global_v_head_offset, int chunk_size, bool use_qk_l2norm,
        float *state_snapshots, int snapshot_stride_floats,
        int max_snapshot_rows)
    {
        if (!merged_qkv || seq_len <= 0 || n_k_heads <= 0 ||
            n_heads <= 0 || d_k <= 0 || d_v <= 0)
        {
            return false;
        }

        const int q_src_dim = n_k_heads * d_k;
        const int k_src_dim = n_k_heads * d_k;
        const int v_dim = n_heads * d_v;
        if (qkv_stride < q_src_dim + k_src_dim + v_dim)
            return false;

        /*
         * Snapshot publication is an output contract, not a distinct recurrence
         * algorithm.  Use the same source view and kernel as ordinary M=1 decode
         * so grouped verifier rows cannot drift into a second arithmetic path.
         */
        const ChunkInputView input{
            .q = merged_qkv,
            .k = merged_qkv + q_src_dim,
            .v = merged_qkv + q_src_dim + k_src_dim,
            .q_row_stride = qkv_stride,
            .k_row_stride = qkv_stride,
            .v_row_stride = qkv_stride,
            .n_k_heads = n_k_heads,
            .global_v_head_offset = global_v_head_offset,
            .layout_name = "merged_qkv"};
        return chunkForwardImpl(
            input, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows,
            InputStateDisposition::PreserveInitialState);
    }

    template <typename RowAccessor>
    bool CPUGatedDeltaNet::chunkForwardBatchedDecodeEquivalentRows(
        RowAccessor &&row_accessor, int value_row_stride,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias, float *output, float *state,
        int seq_len, int request_count, int request_seq_len,
        int n_heads, int d_k, int d_v, bool use_qk_l2norm,
        const int *host_request_seq_lens)
    {
        shared_valid_=false;
        if (!alpha || !beta_raw || !A_log || !dt_bias || !output || !state || !host_request_seq_lens ||
            request_count <= 0 || request_seq_len <= 0 || n_heads <= 0 || d_k <= 0 || d_v <= 0 ||
            d_k > 512 || d_v > 512 || request_seq_len > std::numeric_limits<int>::max()/request_count ||
            seq_len != request_count * request_seq_len)
            return false;
        const CPUGDNStateLayout layout(n_heads, d_k, d_v);
        if (!layout.valid(state) || layout.floats() > std::numeric_limits<int>::max()/request_count ||
            n_heads > std::numeric_limits<int>::max()/request_count) return false;
        int64_t live_rows = 0;
        for (int request = 0; request < request_count; ++request)
        {
            if (host_request_seq_lens[request] < 0 || host_request_seq_lens[request] > request_seq_len) return false;
            live_rows += host_request_seq_lens[request];
        }
        if (verifier_state_capture_ && shared_slots_ &&
            byteRangesOverlap(state,static_cast<size_t>(layout.floats())*sizeof(float),shared_slots_,shared_slot_floats_*sizeof(float))) return false;
        if (!ensureRequestStateBank(request_count, layout.floats(), state)) return false;
        for (int request = 0; request < request_count; ++request)
            if (!layout.valid(request_state_bank_.data() + static_cast<size_t>(request)*layout.floats())) return false;
        const bool capture = verifier_state_capture_ != nullptr;
        if (capture && (!shared_slots_ || verifier_state_capture_!=shared_slots_))
            throw std::invalid_argument("CPU GDN production verifier requires bound shared epoch slots; dense snapshots require explicit serialization API");
        if (capture && (verifier_state_capture_rows_ < seq_len || verifier_state_capture_size_ < layout.floats())) return false;
        const bool shared_capture=capture && shared_slots_;
        if (shared_capture && !prepareShared(request_count,request_seq_len,n_heads,d_k,d_v)) return false;
        ensureScratch(seq_len, n_heads, d_k, d_v);
        const int qk_stride = n_heads*d_k, output_stride = n_heads*d_v;
        const float scale = 1.0f/std::sqrt(static_cast<float>(d_k));
        std::atomic<bool> failed{false};
        PerfStatsCollector::addCounter("kernel", "cpu_gdn_epoch_readout_rows",
            static_cast<double>(live_rows*n_heads), "recurrence", "cpu");
        PerfStatsCollector::addCounter("kernel", "cpu_gdn_request_batched_grouped_calls", 1.0,
            "prefill", "cpu", {{"request_count", std::to_string(request_count)},
            {"request_row_width", std::to_string(request_seq_len)}, {"execution_policy", "request_head_fixed_epoch"}});
        const bool observe_epoch = beginEpochCounterObservations();
        auto &observations = epoch_counter_observations_;
        // Each member keeps its returned local choice through publication;
        // subsequent collector reloads cannot change workshare control flow.
        const auto advance_head = [&](int task) -> CPUFixedEpochObservations
        {
                if (failed.load(std::memory_order_relaxed)) return CPUFixedEpochObservations{};
                const int request=task/n_heads,head=task%n_heads,rows=host_request_seq_lens[request];
                const size_t first=static_cast<size_t>(request)*request_seq_len;
                float *initial=request_state_bank_.data()+static_cast<size_t>(request)*layout.floats()+static_cast<size_t>(head)*layout.headFloats();
                if (shared_capture && rows==0) {
                    const size_t span=CPUSharedGDNSnapshots::headFloats(layout,shared_width_);
                    CPUSharedGDNSnapshots(layout,shared_slots_+static_cast<size_t>(task)*span,shared_width_).retire();
                }
                CPUFixedEpochObservations observed{};
                if (rows > 0)
                {
                    const float *q=nullptr,*k=nullptr,*v=nullptr;
                    row_accessor(request,0,head,q,k,v);
                    const float *prepared_q=q_scratch_.data()+first*qk_stride+head*d_k;
                    const float *prepared_k=k_scratch_.data()+first*qk_stride+head*d_k;
                    float *out=output+first*output_stride+head*d_v;
                    if (shared_capture)
                    {
                        CPUSharedGDNSnapshots shared(layout,shared_slots_+static_cast<size_t>(task)*CPUSharedGDNSnapshots::headFloats(layout,shared_width_),shared_width_);
                        observed = cpu_gdn_epoch::runShared(layout,initial,prepared_q,prepared_k,v,
                            gate_scratch_.data()+first*n_heads+head,beta_sig_scratch_.data()+first*n_heads+head,
                            rows,qk_stride,value_row_stride,n_heads,out,output_stride,shared);
                    }
                    else
                        observed = cpu_gdn_epoch::run(layout,initial,initial,prepared_q,prepared_k,v,
                            gate_scratch_.data()+first*n_heads+head,beta_sig_scratch_.data()+first*n_heads+head,
                            rows,qk_stride,value_row_stride,n_heads,out,output_stride);
                }
                for (int row=rows;row<request_seq_len;++row)
                {
                    std::fill_n(output+(first+row)*output_stride+head*d_v,d_v,0.0f);
                }
                return observed;
        };
        auto work = [&]()
        {
#pragma omp for schedule(static)
            for (int flat = 0; flat < seq_len; ++flat)
            {
                const int request = flat/request_seq_len, row = flat%request_seq_len;
                if (row >= host_request_seq_lens[request]) continue;
                for (int head = 0; head < n_heads; ++head)
                {
                    const float *q=nullptr,*k=nullptr,*v=nullptr;
                    row_accessor(request,row,head,q,k,v);
                    if (!q || !k || !v) { failed.store(true, std::memory_order_relaxed); continue; }
                    float *qd=q_scratch_.data()+static_cast<size_t>(flat)*qk_stride+head*d_k;
                    float *kd=k_scratch_.data()+static_cast<size_t>(flat)*qk_stride+head*d_k;
                    if (use_qk_l2norm) gdn_preprocess_qk_l2norm(q,k,qd,kd,d_k,scale,1e-6f);
                    else gdn_preprocess_qk_scale(q,k,qd,kd,d_k,scale);
                    const size_t index=static_cast<size_t>(flat)*n_heads+head;
                    const float x=alpha[index]+dt_bias[head];
                    gate_scratch_[index]=std::exp(A_log[head]*(x>20.0f ? x : std::log1p(std::exp(x))));
                    beta_sig_scratch_[index]=1.0f/(1.0f+std::exp(-beta_raw[index]));
                }
            }
            if (observe_epoch)
            {
#pragma omp for schedule(static) reduction(cpu_gdn_epoch_observations : observations)
                for (int task = 0; task < request_count*n_heads; ++task)
                    observations += advance_head(task);
            }
            else
            {
#pragma omp for schedule(static)
                for (int task = 0; task < request_count*n_heads; ++task)
                    (void)advance_head(task);
            }
        };
        OMP_WORKSHARE_REGION(work);
        if (observe_epoch) OMP_SINGLE([&]() { publishEpochCounterObservations(d_k, d_v); });
        if (failed.load(std::memory_order_relaxed)) return false;
        if (shared_capture) shared_valid_=true;
        if (!capture) std::memcpy(state,request_state_bank_.data(),static_cast<size_t>(layout.floats())*sizeof(float));
        return true;
    }

    bool CPUGatedDeltaNet::chunkForwardBatchedRequestsWithHostSeqLens(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int request_count, int request_seq_len,
        int n_heads, int d_k, int d_v,
        int chunk_size, bool use_qk_l2norm,
        const int *host_request_seq_lens)
    {
        (void)chunk_size;
        if (!Q || !K || !V)
            return false;

        const int qk_stride = n_heads * d_k;
        const int v_stride = n_heads * d_v;
        auto contiguous_rows =
            [=](int request, int row, int head,
                const float *&q_row,
                const float *&k_row,
                const float *&v_row)
        {
            const int flat_row = request * request_seq_len + row;
            q_row = Q + static_cast<size_t>(flat_row) * qk_stride + head * d_k;
            k_row = K + static_cast<size_t>(flat_row) * qk_stride + head * d_k;
            v_row = V + static_cast<size_t>(flat_row) * v_stride + head * d_v;
        };
        return chunkForwardBatchedDecodeEquivalentRows(
            contiguous_rows, v_stride,
            alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, request_count, request_seq_len,
            n_heads, d_k, d_v, use_qk_l2norm,
            host_request_seq_lens);
    }

    bool CPUGatedDeltaNet::chunkForwardBatchedMergedQKVWithHostSeqLens(
        const float *merged_qkv, int qkv_stride,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int request_count, int request_seq_len,
        int n_k_heads, int n_heads, int d_k, int d_v,
        int global_v_head_offset, bool use_qk_l2norm,
        const int *host_request_seq_lens)
    {
        if (!merged_qkv || n_k_heads <= 0 || n_heads <= 0 ||
            d_k <= 0 || d_v <= 0)
        {
            return false;
        }
        const int q_src_dim = n_k_heads * d_k;
        const int k_src_dim = n_k_heads * d_k;
        const int v_dim = n_heads * d_v;
        if (qkv_stride < q_src_dim + k_src_dim + v_dim)
            return false;

        auto merged_rows =
            [=](int request, int row, int head,
                const float *&q_row,
                const float *&k_row,
                const float *&v_row)
        {
            int qk_head = (head + global_v_head_offset) % n_k_heads;
            if (qk_head < 0)
                qk_head += n_k_heads;
            const int flat_row = request * request_seq_len + row;
            const float *source =
                merged_qkv + static_cast<size_t>(flat_row) * qkv_stride;
            q_row = source + static_cast<size_t>(qk_head) * d_k;
            k_row = source + q_src_dim + static_cast<size_t>(qk_head) * d_k;
            v_row = source + q_src_dim + k_src_dim +
                    static_cast<size_t>(head) * d_v;
        };
        return chunkForwardBatchedDecodeEquivalentRows(
            merged_rows, qkv_stride,
            alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, request_count, request_seq_len,
            n_heads, d_k, d_v, use_qk_l2norm,
            host_request_seq_lens);
    }

    bool CPUGatedDeltaNet::chunkForwardImpl(
        const ChunkInputView &input,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        int /*chunk_size*/, bool use_qk_l2norm,
        float *state_snapshots, int snapshot_stride_floats,
        int max_snapshot_rows,
        InputStateDisposition input_state_disposition)
    {
        shared_valid_=false;
        if (!input.q || !input.k || !input.v || !input.layout_name ||
            input.q_row_stride <= 0 || input.k_row_stride <= 0 ||
            input.v_row_stride <= 0 || input.n_k_heads <= 0 ||
            !alpha || !beta_raw || !A_log || !dt_bias || !output || !state ||
            seq_len <= 0 || n_heads <= 0 || d_k <= 0 || d_v <= 0 ||
            d_k > 512 || d_v > 512)
        {
            return false;
        }

        const CPUGDNStateLayout layout(n_heads, d_k, d_v);
        const size_t state_floats_wide = layout.floats();
        if (!layout.valid(state)) return false;
        if (state_floats_wide >
            static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }
        const int state_floats = static_cast<int>(state_floats_wide);
        const bool capture_state_snapshots = state_snapshots != nullptr;
        const bool shared_capture = capture_state_snapshots && state_snapshots==shared_slots_;
        if (shared_capture) shared_valid_=false;
        if (capture_state_snapshots &&
            (snapshot_stride_floats < state_floats ||
             max_snapshot_rows < seq_len))
        {
            return false;
        }
        if (!capture_state_snapshots &&
            input_state_disposition == InputStateDisposition::PreserveInitialState)
        {
            return false;
        }
        if (capture_state_snapshots)
        {
            const size_t preceding_snapshot_rows =
                static_cast<size_t>(seq_len - 1);
            const size_t snapshot_stride =
                static_cast<size_t>(snapshot_stride_floats);
            if (preceding_snapshot_rows >
                (std::numeric_limits<size_t>::max() - state_floats_wide) /
                    snapshot_stride)
            {
                return false;
            }
            const size_t snapshot_span_floats =
                shared_capture ? shared_slot_floats_ : preceding_snapshot_rows * snapshot_stride + state_floats_wide;
            if (state_floats_wide >
                    std::numeric_limits<size_t>::max() / sizeof(float) ||
                snapshot_span_floats >
                    std::numeric_limits<size_t>::max() / sizeof(float) ||
                byteRangesOverlap(
                    state,
                    state_floats_wide * sizeof(float),
                    state_snapshots,
                    snapshot_span_floats * sizeof(float)))
            {
                return false;
            }
        }

        if (shared_capture && !prepareShared(1,seq_len,n_heads,d_k,d_v)) return false;
        ensureScratch(seq_len, n_heads, d_k, d_v);

        PerfStatsCollector::addCounter("kernel", "cpu_gdn_epoch_readout_rows",
            static_cast<double>(seq_len * n_heads), "recurrence", "cpu");

        const float scale_val = 1.0f / std::sqrt(static_cast<float>(d_k));
        constexpr float l2_eps = 1e-6f;

        // Stride between consecutive timesteps in [seq_len, n_heads * dim] layout
        const int qk_stride = n_heads * d_k;
        const int v_stride = n_heads * d_v;

        const int active_or_requested_workers = omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();
        const int whole_head_worker_count = std::min(active_or_requested_workers, n_heads);

        if (capture_state_snapshots)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_gdn_grouped_verifier_recurrence_calls",
                1.0,
                "verifier",
                "cpu",
                {{"verifier_rows", std::to_string(seq_len)},
                 {"n_heads", std::to_string(n_heads)},
                 {"d_k", std::to_string(d_k)},
                 {"d_v", std::to_string(d_v)},
                 {"snapshot_rows", std::to_string(seq_len)},
                 {"input_layout", input.layout_name},
                 {"parallel_axis", "independent_epoch_heads"},
                 {"active_or_requested_workers", std::to_string(active_or_requested_workers)},
                 {"whole_head_worker_count", std::to_string(whole_head_worker_count)},
                 {"execution_policy", "fixed_epoch_shared_checkpoint"},
                 {"kernel_variant", "epoch8_backward_shared_prefix"},
                 {"snapshot_materialization", shared_capture ? "shared_epoch_checkpoint_raw_log" : "explicit_dense_serialization"},
                 {"arithmetic_order", "serial_decode_per_head"}});
        }
        else
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_gdn_prefill_recurrence_calls",
                1.0,
                seq_len == 1 ? "decode" : "prefill",
                "cpu",
                {{"rows", std::to_string(seq_len)},
                 {"n_heads", std::to_string(n_heads)},
                 {"d_k", std::to_string(d_k)},
                 {"d_v", std::to_string(d_v)},
                 {"input_layout", input.layout_name},
                 {"parallel_axis", "independent_epoch_heads"},
                 {"active_or_requested_workers", std::to_string(active_or_requested_workers)},
                 {"whole_head_worker_count", std::to_string(whole_head_worker_count)},
                 {"arithmetic_order", "serial_decode_per_head"}});
        }

        const bool observe_epoch = beginEpochCounterObservations();
        auto &observations = epoch_counter_observations_;
        // Each member keeps its returned local choice through publication;
        // subsequent collector reloads cannot change workshare control flow.
        const auto advance_head = [&](int head) -> CPUFixedEpochObservations
        {
                const size_t offset = static_cast<size_t>(head) * layout.headFloats();
                float *initial = state + offset;
                float *snapshots = capture_state_snapshots && !shared_capture ? state_snapshots + offset : nullptr;
                if (shared_capture)
                {
                    CPUSharedGDNSnapshots shared(layout,shared_slots_+static_cast<size_t>(head)*CPUSharedGDNSnapshots::headFloats(layout,shared_width_),shared_width_);
                    const auto observed = cpu_gdn_epoch::runShared(layout,initial,
                        q_scratch_.data()+head*d_k,k_scratch_.data()+head*d_k,input.v+head*d_v,
                        gate_scratch_.data()+head,beta_sig_scratch_.data()+head,
                        seq_len,qk_stride,input.v_row_stride,n_heads,output+head*d_v,v_stride,shared);
                    if (input_state_disposition==InputStateDisposition::PublishTerminalState)
                        shared.materialize(seq_len-1,initial);
                    return observed;
                }
                else
                {
                    float *working=capture_state_snapshots
                        ? snapshots+static_cast<size_t>(seq_len-1)*snapshot_stride_floats : initial;
                    const auto observed = cpu_gdn_epoch::run(layout,initial,working,
                        q_scratch_.data()+head*d_k,k_scratch_.data()+head*d_k,input.v+head*d_v,
                        gate_scratch_.data()+head,beta_sig_scratch_.data()+head,
                        seq_len,qk_stride,input.v_row_stride,n_heads,output+head*d_v,v_stride,
                        snapshots,snapshot_stride_floats);
                    if (capture_state_snapshots && input_state_disposition==InputStateDisposition::PublishTerminalState)
                        std::memcpy(initial,working,static_cast<size_t>(layout.headFloats())*sizeof(float));
                    return observed;
                }
        };
        auto do_work = [&]()
        {
        // ── Phase 1: Parallel preprocessing (across tokens) ──────────
#pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < n_heads; ++h)
                {
                    const int qk_off = t * qk_stride + h * d_k;
                    float *q_dst = q_scratch_.data() + qk_off;
                    float *k_dst = k_scratch_.data() + qk_off;
                    if (h >= input.n_k_heads)
                    {
                        // Modular source-head mapping repeats after n_k_heads,
                        // including nonzero/negative offsets. The same token's
                        // earlier head owns the already rounded normalization;
                        // copying it avoids repeating reductions and square roots.
                        // This whole token belongs to one worker, so no event or
                        // additional worksharing barrier is required here.
                        const int prepared_off = t * qk_stride + (h % input.n_k_heads) * d_k;
                        std::memcpy(q_dst, q_scratch_.data() + prepared_off, d_k * sizeof(float));
                        std::memcpy(k_dst, k_scratch_.data() + prepared_off, d_k * sizeof(float));
                    }
                    else
                    {
                        int source_head =
                            (h + input.global_v_head_offset) % input.n_k_heads;
                        if (source_head < 0)
                            source_head += input.n_k_heads;
                        const float *q_src =
                            input.q + static_cast<size_t>(t) * input.q_row_stride +
                            static_cast<size_t>(source_head) * d_k;
                        const float *k_src =
                            input.k + static_cast<size_t>(t) * input.k_row_stride +
                            static_cast<size_t>(source_head) * d_k;

                        if (use_qk_l2norm)
                        {
                            gdn_preprocess_qk_l2norm(q_src, k_src, q_dst, k_dst, d_k, scale_val, l2_eps);
                        }
                        else
                        {
                            gdn_preprocess_qk_scale(q_src, k_src, q_dst, k_dst, d_k, scale_val);
                        }
                    }

                    // Compute the exact decode gate expressions while tokens
                    // are already distributed across the OpenMP team. The old
                    // AVX fast-exp batch changed arithmetic between prefill and
                    // decode; keeping these canonical expressions here both
                    // preserves parallelism and removes a second head traversal.
                    const int gi = t * n_heads + h;
                    const float x = alpha[gi] + dt_bias[h];
                    const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
                    gate_scratch_[gi] = std::exp(A_log[h] * sp);
                    beta_sig_scratch_[gi] =
                        1.0f / (1.0f + std::exp(-beta_raw[gi]));

                    // Projection owns and overwrites every live output value.
                }

            }
            // implicit barrier between omp-for regions

            // Each head owns one canonical checkpoint and ordered raw log.
            // Groups share checkpoint actions; only fixed epoch boundaries flush.
            if (observe_epoch)
            {
#pragma omp for schedule(static) reduction(cpu_gdn_epoch_observations : observations)
                for (int head = 0; head < n_heads; ++head)
                    observations += advance_head(head);
            }
            else
            {
#pragma omp for schedule(static)
                for (int head = 0; head < n_heads; ++head)
                    (void)advance_head(head);
            }
        };
        /*
         * Use the stable process-wide team even when the number of recurrent
         * heads is smaller than the socket width. A bounded team looks cheaper
         * in an isolated kernel sample, but libgomp may retire every omitted
         * worker and make the following full-width projection recreate it.
         * The omp-for loops naturally leave workers without a head idle while
         * preserving the process-wide team for the next captured CPU stage.
         */
        OMP_WORKSHARE_REGION(do_work);
        if (observe_epoch) OMP_SINGLE([&]() { publishEpochCounterObservations(d_k, d_v); });
        if (shared_capture) shared_valid_=true;

#if defined(__AVX512F__)
#undef GDN_PREFETCH_S
#endif

        return true;
    }

} // namespace llaminar2
