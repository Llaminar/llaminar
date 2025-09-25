// COSMA Prefill Manager (Phase 1 / 1b Implementation)
// See plan: .github/instructions/cosma-prefill-plan.instructions.md
// Scope implemented here: gating, strategy cache, activation/weight streaming,
// small-op fast path, validation tile, cumulative memory budget guard,
// instrumentation counters & log level mapping.
// Additional debug: LLAMINAR_COSMA_DUMP_SMALL -> dump ownership & values for tiny matrices (<=8x8)
// Deferred (future phases): fused dequant->layout, in-layout elementwise kernels,
// overlap of stream + GEMM.
#include "cosma_prefill_manager.h"
#include "logger.h"
#include "quant_dequant.h"
#include <cosma/multiply.hpp>
#include <cosma/cinterface.hpp>
#include <cosma/mapper.hpp>
#include <cstdlib>
#include <cstring>
#include <cstring>
#include <algorithm>
#include <cblas.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <random>
#include <future>
#include <limits>
#include <cmath>
#include <cctype>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    inline bool mpi_is_initialized()
    {
        int flag = 0;
        MPI_Initialized(&flag);
        return flag != 0;
    }
    inline bool mpi_is_finalized()
    {
        int flag = 0;
        MPI_Finalized(&flag);
        return flag != 0;
    }
    inline int mpi_world_size_safe()
    {
        if (!mpi_is_initialized() || mpi_is_finalized())
            return 1;
        int sz = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &sz);
        return sz;
    }
    inline int mpi_rank_safe()
    {
        if (!mpi_is_initialized() || mpi_is_finalized())
            return 0;
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
    }
    inline void safe_allreduce_float_inplace(float *buf, int count, MPI_Op op)
    {
        if (mpi_is_initialized() && mpi_world_size_safe() > 1)
            MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_FLOAT, op, MPI_COMM_WORLD);
    }
    inline void safe_allreduce_int_inplace(int *buf, int count, MPI_Op op)
    {
        if (mpi_is_initialized() && mpi_world_size_safe() > 1)
            MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_INT, op, MPI_COMM_WORLD);
    }
    inline void safe_bcast(void *buf, int count, MPI_Datatype dt, int root)
    {
        if (mpi_is_initialized() && mpi_world_size_safe() > 1)
            MPI_Bcast(buf, count, dt, root, MPI_COMM_WORLD);
    }
    inline void safe_barrier()
    {
        if (mpi_is_initialized() && mpi_world_size_safe() > 1)
            MPI_Barrier(MPI_COMM_WORLD);
    }
}

namespace llaminar
{

    FusedRmsnormQkvResult::WeightGuard::WeightGuard(CosmaPrefillManager *mgr, CosmaWeightHandle &&h)
        : manager(mgr), handle(std::move(h))
    {
    }

    FusedRmsnormQkvResult::WeightGuard::WeightGuard(WeightGuard &&other) noexcept
        : manager(other.manager), handle(std::move(other.handle))
    {
        other.manager = nullptr;
    }

    FusedRmsnormQkvResult::WeightGuard &FusedRmsnormQkvResult::WeightGuard::operator=(WeightGuard &&other) noexcept
    {
        if (this != &other)
        {
            if (manager)
            {
                manager->release_weight(std::move(handle));
            }
            manager = other.manager;
            handle = std::move(other.handle);
            other.manager = nullptr;
        }
        return *this;
    }

    FusedRmsnormQkvResult::WeightGuard::~WeightGuard()
    {
        if (manager)
        {
            std::cerr << "[CosmaPrefill][weight-guard] releasing " << handle.desc.id << std::endl;
            manager->release_weight(std::move(handle));
        }
    }

    std::string StrategyCache::make_key(int m, int n, int k, int p) const
    {
        return std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) + ":p=" + std::to_string(p);
    }

    const cosma::Strategy &StrategyCache::get(int m, int n, int k, int p)
    {
        auto key = make_key(m, n, k, p);
        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            stats.hits++;
            return it->second;
        }
        // Construct new strategy (simple default for now) with diagnostic logging
        LOG_DEBUG("[StrategyCache] constructing strategy key=" << key << " m=" << m << " n=" << n << " k=" << k << " p=" << p);
        cosma::Strategy strat(m, n, k, p);
        LOG_DEBUG("[StrategyCache] constructed strategy key=" << key);
        auto res = cache_.emplace(key, std::move(strat));
        stats.misses++;
        return res.first->second;
    }

    CosmaPrefillManager &CosmaPrefillManager::instance()
    {
        static CosmaPrefillManager inst;
        return inst;
    }

    CosmaPrefillManager::~CosmaPrefillManager()
    {
        try
        {
            dump_stats_if_requested();
        }
        catch (...)
        {
            // Destructors must not throw; ignore failures on shutdown.
        }
    }

    CosmaPrefillManager::CosmaPrefillManager()
    {
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        if (std::getenv("LLAMINAR_SKIP_MPI_IN_SINGLE_TEST"))
        {
            world_size_ = 1;
            rank_ = 0;
        }
        else if (mpi_init)
        {
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        }
        else
        {
            world_size_ = 1;
            rank_ = 0;
        }
        // Environment initialization
        if (const char *env_t = std::getenv("LLAMINAR_COSMA_PREFILL_THRESHOLD"))
        {
            threshold_ = std::atoi(env_t);
        }
        if (const char *env_fast = std::getenv("LLAMINAR_COSMA_FAST_PATH_THRESHOLD"))
        {
            long long v = std::atoll(env_fast);
            if (v > 0)
                fast_path_threshold_ops_ = v;
        }
        if (const char *env_val = std::getenv("LLAMINAR_COSMA_VALIDATE_TILE"))
        {
            validate_tile_tokens_ = std::atoi(env_val);
        }
        if (const char *env_log = std::getenv("LLAMINAR_COSMA_LOG_LEVEL"))
        {
            std::string lvl(env_log);
            if (lvl == "error")
                log_level_ = 0;
            else if (lvl == "warn")
                log_level_ = 1;
            else if (lvl == "info")
                log_level_ = 2;
            else if (lvl == "debug")
                log_level_ = 3;
            else if (lvl == "trace")
                log_level_ = 4;
        }
        // Escalate to TRACE automatically when diagnostics or test trace enabled
        if (std::getenv("LLAMINAR_COSMA_DIAG") || std::getenv("LLAMINAR_COSMA_TEST_TRACE"))
        {
            log_level_ = 4; // TRACE
            // Also escalate global logger so LOG_TRACE macros emit output
            Logger::getInstance().setLogLevel(LogLevel::TRACE);
        }
        if (const char *env_mem = std::getenv("LLAMINAR_COSMA_MAX_RESIDENT_MB"))
        {
            long long v = std::atoll(env_mem);
            if (v > 0)
                max_resident_mb_ = v;
        }
        if (std::getenv("LLAMINAR_COSMA_FORCE"))
        {
            force_cosma_ = true;
        }
    }

    CosmaPrefillManager::EnvSnapshot CosmaPrefillManager::capture_env_snapshot() const
    {
        EnvSnapshot env;
        env.cosma_disabled = std::getenv("LLAMINAR_COSMA_DISABLE") != nullptr;
        env.adaptive_disabled = std::getenv("ADAPTIVE_DISABLE_COSMA") != nullptr;
        env.diag_enabled = std::getenv("LLAMINAR_COSMA_DIAG") != nullptr;
        env.diag_deep = std::getenv("LLAMINAR_COSMA_DIAG_DEEP") != nullptr;
        env.diag_axis = std::getenv("LLAMINAR_COSMA_DIAG_AXIS") != nullptr;
        env.diag_coord_invert = std::getenv("LLAMINAR_COSMA_DIAG_COORD_INVERT") != nullptr;
        env.diag_local_probe = std::getenv("LLAMINAR_COSMA_DIAG_LOCAL_PROBE") != nullptr;
        env.diag_local_probe_deep = std::getenv("LLAMINAR_COSMA_DIAG_LOCAL_PROBE_DEEP") != nullptr;
        env.diag_recon_bypass = std::getenv("LLAMINAR_COSMA_DIAG_RECON_BYPASS") != nullptr;
        env.diag_recon_transpose = std::getenv("LLAMINAR_COSMA_DIAG_RECON_TRANSPOSE") != nullptr;
        env.diag_recon_brute = std::getenv("LLAMINAR_COSMA_DIAG_RECON_BRUTE") != nullptr;
        env.diag_recon_map = std::getenv("LLAMINAR_COSMA_DIAG_RECON_MAP") != nullptr;
        env.recon_force_legacy = std::getenv("LLAMINAR_COSMA_RECON_FORCE_LEGACY") != nullptr;
        env.diag_swaprc = std::getenv("LLAMINAR_COSMA_DIAG_SWAPRC") != nullptr;
        env.diag_try_transpose = std::getenv("LLAMINAR_COSMA_DIAG_TRY_TRANSPOSE") != nullptr;
        env.diag_skip_norm = std::getenv("LLAMINAR_COSMA_DIAG_SKIP_NORM") != nullptr;
        env.debug_recon = std::getenv("LLAMINAR_COSMA_DEBUG_RECON") != nullptr;
        env.compare_replicated = std::getenv("LLAMINAR_COSMA_COMPARE_REPLICATED") != nullptr;
        env.diag_dump_small = std::getenv("LLAMINAR_COSMA_DUMP_SMALL") != nullptr;
        env.pop_forward_legacy = std::getenv("LLAMINAR_COSMA_POP_FORWARD_LEGACY") != nullptr;
        env.force_distributed_act = std::getenv("LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT") != nullptr;
        env.fast_unverified = std::getenv("LLAMINAR_COSMA_FAST_UNVERIFIED") != nullptr;
        env.disable_fused_dequant = std::getenv("LLAMINAR_COSMA_DISABLE_FUSED_DEQUANT") != nullptr;
        env.force_replicated_diag = std::getenv("LLAMINAR_COSMA_FORCE_REPLICATED_DIAG") != nullptr;
        env.force_replicated = std::getenv("LLAMINAR_COSMA_FORCE_REPLICATED") != nullptr;
        env.force_direct = std::getenv("LLAMINAR_COSMA_FORCE_DIRECT") != nullptr;
        env.replicate_B = std::getenv("LLAMINAR_COSMA_REPLICATE_B") != nullptr;
        env.auto_fix_transpose = std::getenv("LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE") != nullptr;
        env.force_unified_strategy = std::getenv("LLAMINAR_COSMA_FORCE_UNIFIED") != nullptr;
        env.overlap_enabled = std::getenv("LLAMINAR_COSMA_OVERLAP_STREAM") != nullptr;
        env.overlap_verbose = std::getenv("LLAMINAR_COSMA_OVERLAP_VERBOSE") != nullptr;
        env.preflight_disable = std::getenv("LLAMINAR_COSMA_PREFLIGHT_DISABLE") != nullptr;
        if (const char *direct_env = std::getenv("LLAMINAR_COSMA_DIRECT_THRESHOLD_OPS"))
        {
            long long parsed = std::atoll(direct_env);
            if (parsed > 0)
            {
                env.direct_threshold_override = true;
                env.direct_threshold_ops = parsed;
            }
        }
        if (env.diag_enabled)
        {
            if (const char *tap = std::getenv("LLAMINAR_COSMA_DIAG_TAP"))
            {
                env.diag_tap_enabled = true;
                env.diag_tap_value = std::max(1, std::atoi(tap));
            }
        }
        if (const char *perm = std::getenv("LLAMINAR_COSMA_DIAG_PERM_INFER"))
        {
            env.diag_perm_infer_active = perm[0] != '\0';
            if (env.diag_perm_infer_active)
                env.diag_perm_spec = perm;
        }
        if (const char *samples = std::getenv("LLAMINAR_COSMA_DIAG_SAMPLES"))
        {
            env.diag_samples_active = samples[0] != '\0';
            if (env.diag_samples_active)
                env.diag_samples_spec = samples;
        }
        if (const char *safety = std::getenv("LLAMINAR_COSMA_PREFLIGHT_SAFETY_FACTOR"))
        {
            try
            {
                env.preflight_safety = std::stod(safety);
                env.preflight_safety = std::max(0.5, std::min(4.0, env.preflight_safety));
                env.preflight_safety_override = true;
            }
            catch (...)
            {
                env.preflight_safety = 1.2;
            }
        }
        if (const char *v = std::getenv("LLAMINAR_OPENBLAS_THREADS"))
        {
            env.forced_openblas_threads = std::max(1, std::atoi(v));
        }
        else if (const char *v = std::getenv("OPENBLAS_NUM_THREADS"))
        {
            env.forced_openblas_threads = std::max(1, std::atoi(v));
        }
        if (const char *v = std::getenv("LLAMINAR_COSMA_FORCE_REPLICATED_THREADS"))
        {
            env.forced_replicated_threads = std::max(1, std::atoi(v));
        }
        return env;
    }

    const CosmaPrefillManager::EnvSnapshot &CosmaPrefillManager::resolve_env(const EnvSnapshot *override_env, EnvSnapshot &storage) const
    {
        if (override_env)
        {
            return *override_env;
        }
        storage = capture_env_snapshot();
        return storage;
    }

    CosmaView CosmaPrefillManager::convert_activation_in(const float *row_major, int m, int k)
    {
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        if (world_size_ == 1)
        {
            CosmaView v;
            v.global_rows = m;
            v.global_cols = k;
            v.label = 'A';
            v.original_row_major = row_major;
            return v;
        }
        // Budget check: activation bytes + safety factor (assume float32)
        size_t act_bytes = (size_t)m * k * sizeof(float);
        if (!memory_budget_allows(act_bytes))
        {
            if (rank_ == 0 && should_log(1))
            {
                LOG_WARN("[CosmaPrefill] Memory budget denial for activation convert bytes=" << act_bytes);
            }
            CosmaView v;
            v.global_rows = m;
            v.global_cols = k;
            v.label = 'A';
            v.original_row_major = row_major;
            return v; // fallback (will trigger single/fast path)
        }
        auto &strat = strategy_cache_.get(m, k, k, world_size_);
        auto view = allocate_matrix('A', m, k, strat, false);
        fill_activation(view, row_major, m, k, env);
        view.original_row_major = row_major;
        return view;
    }

    bool CosmaPrefillManager::enabled_for(int seq_len) const
    {
        EnvSnapshot storage;
        const auto &env = resolve_env(nullptr, storage);
        if (env.cosma_disabled || env.adaptive_disabled)
            return false;
        if (force_cosma_)
            return true;
        return world_size_ > 1 && seq_len >= threshold_;
    }

    CosmaView CosmaPrefillManager::allocate_matrix(char label, int m, int n, const cosma::Strategy &strat, bool zero)
    {
        CosmaView v;
        v.global_rows = m;
        v.global_cols = n;
        v.label = label;
        try
        {
            auto deleter = [label](cosma::CosmaMatrix<float> *ptr)
            {
                if (ptr)
                {
                    auto data_ptr = static_cast<void *>(ptr->matrix_pointer());
                    std::cerr << "[CosmaPrefill][deallocate_matrix] label=" << label
                              << " ptr=" << data_ptr << std::endl;
                }
                delete ptr;
            };
            std::shared_ptr<cosma::CosmaMatrix<float>> mat(new cosma::CosmaMatrix<float>(label, strat, rank_, false), deleter);
            // Phase 1b correction: we must allocate here so that fill_activation / stream_weight_blocks
            // can populate data; previously skipping allocation led to uninitialized distributed GEMM inputs.
            try
            {
                mat->allocate();
            }
            catch (...)
            {
            }
            std::cerr << "[CosmaPrefill][allocate_matrix] label=" << label << " m=" << m << " n=" << n
                      << " ptr=" << static_cast<void *>(mat->matrix_pointer()) << std::endl;
            if (zero && mat->matrix_pointer())
            {
                std::fill(mat->matrix_pointer(), mat->matrix_pointer() + mat->matrix_size(), 0.f);
            }
            v.mat = mat;
            // Phase 1b: memory tracking if storage already materialized
            if (mat->matrix_pointer())
            {
                std::lock_guard<std::mutex> lock(allocations_mutex_);
                // Avoid double-counting: ensure this exact shared_ptr not already tracked
                bool already = false;
                for (auto &rec : allocations_)
                    if (!rec.ref.expired() && rec.ref.lock().get() == mat.get())
                    {
                        already = true;
                        break;
                    }
                if (!already)
                {
                    long long bytes = (long long)mat->matrix_size() * (long long)sizeof(float);
                    stats_.current_resident_bytes += bytes;
                    long long cur = stats_.current_resident_bytes.load();
                    long long peak = stats_.peak_resident_bytes.load();
                    while (cur > peak && !stats_.peak_resident_bytes.compare_exchange_weak(peak, cur))
                    {
                        peak = stats_.peak_resident_bytes.load();
                    }
                    stats_.allocations_tracked++;
                    allocations_.push_back({mat, bytes});
                }
            }
        }
        catch (const std::exception &e)
        {
            if (rank_ == 0 && should_log(0))
            {
                LOG_ERROR("[CosmaPrefill] allocate_matrix exception label=" << label << " m=" << m << " n=" << n << " : " << e.what());
            }
        }
        return v;
    }

    void CosmaPrefillManager::scatter_row_major_dest_local(CosmaView &dst, const float *src_row_major, int rows, int cols)
    {
        if (!dst.mat || !src_row_major)
            return;
        float *local = dst.mat->matrix_pointer();
        size_t sz = dst.mat->matrix_size();
        if (!local || sz == 0)
            return;
        std::fill(local, local + sz, 0.f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long li = 0; li < static_cast<long long>(sz); ++li)
        {
            auto gc = dst.mat->global_coordinates(static_cast<int>(li));
            int gi = gc.first;
            int gj = gc.second;
            if (gi >= 0 && gi < rows && gj >= 0 && gj < cols)
            {
                local[li] = src_row_major[(size_t)gi * cols + gj];
            }
        }
    }

    void CosmaPrefillManager::fill_activation(CosmaView &dst, const float *src_row_major, int m, int k, const EnvSnapshot &env)
    {
        if (!dst.mat || !src_row_major)
            return;
        float *local = dst.mat->matrix_pointer();
        size_t sz = dst.mat->matrix_size();
        if (!local || sz == 0)
            return;
        bool trace_scatter = env.diag_deep && should_log(4);
        // Destination-local population is now the default (proved correct); legacy forward mode only if explicitly forced.
        bool legacy_forward = env.pop_forward_legacy;
        bool dest_local_mode = !legacy_forward; // default true
        if (dest_local_mode)
        {
            if (legacy_forward && rank_ == 0 && should_log(1))
            {
                LOG_WARN("[CosmaPrefill][pop] legacy forward population unexpectedly active while dest-local enabled");
            }
            else if (trace_scatter && rank_ == 0)
            {
                LOG_TRACE("[CosmaPrefill][pop] destination-local population (A) default");
            }
            scatter_row_major_dest_local(dst, src_row_major, m, k);
            if (trace_scatter && rank_ == 0)
            {
                for (int gi = 0; gi < std::min(m, 8); ++gi)
                {
                    for (int gj = 0; gj < std::min(k, 8); ++gj)
                    {
                        auto lc = dst.mat->local_coordinates(gi, gj);
                        int local_index = lc.first;
                        int owner = lc.second;
                        if (owner == rank_ && local_index >= 0 && (size_t)local_index < sz)
                        {
                            LOG_TRACE("[CosmaPrefill][scatterA-dest] li=" << local_index << " (gi=" << gi << ",gj=" << gj << ") val=" << local[local_index]);
                        }
                    }
                }
            }
        }
        else
        {
            std::fill(local, local + sz, 0.f);
            int trace_rows = std::min(m, 8);
            int trace_cols = std::min(k, 8);
            for (int gi = 0; gi < m; ++gi)
            {
                const float *row_ptr = src_row_major + (size_t)gi * k;
                for (int gj = 0; gj < k; ++gj)
                {
                    auto lc = dst.mat->local_coordinates(gi, gj);
                    int local_index = lc.first;
                    int owner = lc.second;
                    if (owner == rank_ && local_index >= 0 && (size_t)local_index < sz)
                    {
                        local[local_index] = row_ptr[gj];
                        if (trace_scatter && gi < trace_rows && gj < trace_cols && rank_ == 0)
                        {
                            LOG_TRACE("[CosmaPrefill][scatterA] (gi=" << gi << ",gj=" << gj << ") val=" << row_ptr[gj] << " lidx=" << local_index);
                        }
                        if (env.diag_coord_invert && gi < 128 && gj < 8)
                        {
                            auto gc = dst.mat->global_coordinates(local_index);
                            if ((gc.first != gi || gc.second != gj) && rank_ == 0 && should_log(2))
                            {
                                LOG_INFO("[CosmaPrefill][coord-check] mismatch li=" << local_index << " stored=(gi=" << gi << ",gj=" << gj
                                                                                    << ") global_coordinates -> (" << gc.first << "," << gc.second << ")");
                            }
                        }
                    }
                }
            }
        }
        // Local index probe (population correctness) for A
        if (env.diag_local_probe && rank_ == 0)
        {
            size_t sample = std::min(sz, (size_t)8192);
            long double num = 0.0L, den = 0.0L;
            size_t mismatches = 0;
            int logged = 0;
            for (size_t li = 0; li < sample; ++li)
            {
                auto gc = dst.mat->global_coordinates((int)li);
                int gi = gc.first;
                int gj = gc.second;
                if (gi < 0 || gj < 0 || gi >= m || gj >= k)
                    continue; // skip invalid
                float v = local[li];
                float ref = src_row_major[(size_t)gi * k + gj];
                long double diff = (long double)v - (long double)ref;
                num += diff * diff;
                den += (long double)ref * (long double)ref;
                if (v != ref && logged < 8)
                {
                    LOG_INFO("[CosmaPrefill][local-probe] A mismatch li=" << li << " -> (gi=" << gi << ",gj=" << gj << ") v=" << v << " ref=" << ref);
                    ++logged;
                    ++mismatches;
                }
                else if (v != ref)
                {
                    ++mismatches;
                }
            }
            double rel = (den > 0) ? (double)(std::sqrt(num) / (std::sqrt(den) + 1e-30L)) : 0.0;
            LOG_INFO("[CosmaPrefill][local-probe] A prefix_rel_l2=" << rel << " sample=" << sample << " mismatches=" << mismatches);
        }
        stats_.bytes_converted_activations += (long long)m * k * (long long)sizeof(float);
    }

    CosmaView CosmaPrefillManager::convert_activation_in_with_strategy(const float *row_major, int m, int k, const cosma::Strategy &strat)
    {
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        // Fast-path gating: if total matmul volume would fall below fast_path_threshold_ops_,
        // avoid allocating COSMA matrix (we'll use original row-major pointers in matmul fast path).
        // For multi-rank elementwise validation we sometimes need to force allocation even for small tensors.
        long long prospective_volume = 1ll * m * k;                                                                                                                  // partial (full volume depends on n, but conservative cheap check)
        bool skip_cosma = !env.force_distributed_act && ((world_size_ == 1) || (fast_path_threshold_ops_ > 0 && prospective_volume < fast_path_threshold_ops_ / 8)); // heuristic
        if (skip_cosma)
        {
            CosmaView v;
            v.global_rows = m;
            v.global_cols = k;
            v.label = 'A';
            v.original_row_major = row_major;
            return v;
        }
        auto view = allocate_matrix('A', m, k, strat, false);
        fill_activation(view, row_major, m, k, env);
        view.original_row_major = row_major;
        return view;
    }

    CosmaView CosmaPrefillManager::convert_activation_operand(const float *row_major, int m, int k, const cosma::Strategy &strat)
    {
        // Force usage of C-strategy for all operands to keep mapper coherent.
        return convert_activation_in_with_strategy(row_major, m, k, strat);
    }

    CosmaWeightHandle CosmaPrefillManager::load_weight_operand(const WeightDescriptor &desc, const cosma::Strategy &strat)
    {
        return load_weight_with_strategy(desc, strat);
    }

    void CosmaPrefillManager::stream_weight_blocks(CosmaView &dst, const WeightDescriptor &desc, const EnvSnapshot &env)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        const float *base = static_cast<const float *>(desc.base_ptr);
        if (!dst.mat || !base)
            return;
        size_t sz = dst.mat->matrix_size();
        size_t total = static_cast<size_t>(desc.rows) * desc.cols;
        const bool needs_copy = (desc.row_stride > 0 && desc.row_stride != desc.cols) ||
                                (desc.col_stride > 0 && desc.col_stride != 1);
        std::vector<float> scratch;
        const float *row_major = base;
        if (needs_copy)
        {
            scratch.resize(total, 0.f);
            for (int gi = 0; gi < desc.rows; ++gi)
            {
                const float *row_ptr = base + (size_t)gi * (desc.row_stride > 0 ? desc.row_stride : desc.cols);
                for (int gj = 0; gj < desc.cols; ++gj)
                {
                    scratch[(size_t)gi * desc.cols + gj] = row_ptr[gj * (desc.col_stride > 0 ? desc.col_stride : 1)];
                }
            }
            row_major = scratch.data();
        }
        scatter_row_major_dest_local(dst, row_major, desc.rows, desc.cols);
        stats_.bytes_streamed_weights += static_cast<long long>(std::min(sz, total) * sizeof(float));
        auto t1 = std::chrono::high_resolution_clock::now();
        stats_.us_stream_weights += (long long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }

    void CosmaPrefillManager::stream_weight_blocks_quantized(CosmaView &dst, const WeightDescriptor &desc, const EnvSnapshot &env)
    {
        // Phase 2 fused path: directly dequantize each block and scatter owned elements
        // Supported: quant_type==1 (synthetic Q5_0). Fallback otherwise or when disabled.
        if (env.disable_fused_dequant || desc.quant_type != 1 || !dst.mat || !desc.base_ptr)
        {
            stream_weight_blocks(dst, desc, env);
            return;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        const size_t total = static_cast<size_t>(desc.rows) * desc.cols;
        const int block_vals = 32; // Q5_0 block size
        const size_t n_blocks = (total + block_vals - 1) / block_vals;
        const uint8_t *qptr = static_cast<const uint8_t *>(desc.base_ptr);
        std::vector<float> scratch(total, 0.f);
        std::vector<float> tmp(block_vals, 0.f);
        bool keep_row_major = env.debug_recon || env.compare_replicated;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            llaminar::dequant_block_q5_0(qptr, tmp.data(), block_vals);
            size_t base_index = b * block_vals;
            size_t remain = std::min<size_t>(block_vals, total - base_index);
            // Scatter the portion of this block that maps to each (row,col)
            for (size_t i = 0; i < remain; ++i)
            {
                size_t linear = base_index + i;
                scratch[linear] = tmp[i];
            }
            qptr += desc.quant_block_size ? desc.quant_block_size : (size_t)(2 + 4 + 16);
        }
        scatter_row_major_dest_local(dst, scratch.data(), desc.rows, desc.cols);
        dst.host_owned = std::make_shared<std::vector<float>>(std::move(scratch));
        dst.original_row_major = dst.host_owned->data();
        stats_.bytes_streamed_weights += static_cast<long long>(total * sizeof(float));
        stats_.fused_dequant_invocations++;
        stats_.fused_dequant_elements += static_cast<long long>(total);
        auto t1 = std::chrono::high_resolution_clock::now();
        stats_.us_stream_weights += (long long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }

    CosmaWeightHandle CosmaPrefillManager::load_weight(const WeightDescriptor &desc)
    {
        if (world_size_ == 1)
        {
            CosmaView v;
            v.global_rows = desc.rows;
            v.global_cols = desc.cols;
            v.label = 'B';
            v.original_row_major = static_cast<const float *>(desc.base_ptr);
            return {v, desc};
        }
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        size_t w_bytes = (size_t)desc.rows * desc.cols * sizeof(float);
        if (!memory_budget_allows(w_bytes))
        {
            if (rank_ == 0 && should_log(1))
            {
                LOG_WARN("[CosmaPrefill] Memory budget denial for weight load bytes=" << w_bytes << " id=" << desc.id);
            }
            CosmaView v;
            v.global_rows = desc.rows;
            v.global_cols = desc.cols;
            v.label = 'B';
            v.original_row_major = static_cast<const float *>(desc.base_ptr);
            return {v, desc};
        }
        auto &strat = strategy_cache_.get(desc.rows, desc.cols, desc.rows, world_size_);
        auto view = allocate_matrix('B', desc.rows, desc.cols, strat, false);
        if (desc.quant_type > 0)
        {
            stream_weight_blocks_quantized(view, desc, env);
        }
        else
        {
            stream_weight_blocks(view, desc, env);
        }
        if (desc.quant_type == 0 && !view.original_row_major)
        {
            view.original_row_major = static_cast<const float *>(desc.base_ptr);
        }
        return CosmaWeightHandle{view, desc};
    }

    CosmaWeightHandle CosmaPrefillManager::load_weight_with_strategy(const WeightDescriptor &desc, const cosma::Strategy &strat)
    {
        long long prospective_volume = 1ll * desc.rows * desc.cols;
        bool skip_cosma = (world_size_ == 1) || (fast_path_threshold_ops_ > 0 && prospective_volume < fast_path_threshold_ops_ / 8);
        if (skip_cosma)
        {
            CosmaView v;
            v.global_rows = desc.rows;
            v.global_cols = desc.cols;
            v.label = 'B';
            v.original_row_major = static_cast<const float *>(desc.base_ptr);
            return {v, desc};
        }
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        size_t w_bytes = (size_t)desc.rows * desc.cols * sizeof(float);
        if (!memory_budget_allows(w_bytes))
        {
            if (rank_ == 0 && should_log(1))
            {
                LOG_WARN("[CosmaPrefill] Memory budget denial for weight load (explicit strategy) bytes=" << w_bytes << " id=" << desc.id);
            }
            CosmaView v;
            v.global_rows = desc.rows;
            v.global_cols = desc.cols;
            v.label = 'B';
            v.original_row_major = static_cast<const float *>(desc.base_ptr);
            return {v, desc};
        }
        auto view = allocate_matrix('B', desc.rows, desc.cols, strat, false);
        if (desc.quant_type > 0)
        {
            stream_weight_blocks_quantized(view, desc, env);
        }
        else
        {
            stream_weight_blocks(view, desc, env);
        }
        if (desc.quant_type == 0 && !view.original_row_major)
        {
            view.original_row_major = static_cast<const float *>(desc.base_ptr);
        }
        return CosmaWeightHandle{view, desc};
    }

    CosmaView CosmaPrefillManager::matmul(const CosmaView &A,
                                          const CosmaWeightHandle &W,
                                          int m, int k, int n,
                                          bool transposeW,
                                          float alpha,
                                          float beta)
    {
        if (rank_ == 0 && should_log(3))
        {
            LOG_DEBUG("[CosmaPrefill][matmul] enter m=" << m << " n=" << n << " k=" << k << " transposeW=" << transposeW);
        }
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        const bool dump_small_enabled = env.diag_dump_small;
        const bool diag_enabled = env.diag_enabled;
        const bool diag_tap_enabled = env.diag_tap_enabled;
        const int diag_tap_value = env.diag_tap_value;
        const bool force_replicated_diag = env.force_replicated_diag;
        const bool force_large_replicated = env.force_replicated;
        const bool cosma_disabled = env.adaptive_disabled || env.cosma_disabled;
        long long direct_threshold_ops = fast_path_threshold_ops_;
        if (env.direct_threshold_override)
            direct_threshold_ops = env.direct_threshold_ops;
        const bool force_direct = env.force_direct;
        const bool replicate_B = env.replicate_B;
        const bool diag_deep = env.diag_deep;
        const bool diag_local_probe = env.diag_local_probe;
        const bool diag_local_probe_deep = env.diag_local_probe_deep;
        const bool auto_fix_transpose = env.auto_fix_transpose;
        const bool force_unified_strategy = env.force_unified_strategy;
        const bool pop_forward_legacy = env.pop_forward_legacy;
        const bool diag_recon_bypass = env.diag_recon_bypass;
        const bool diag_recon_transpose = env.diag_recon_transpose;
        const bool diag_swaprc = env.diag_swaprc;
        const char *diag_perm_spec = env.diag_perm_infer_active ? env.diag_perm_spec.c_str() : nullptr;
        const bool diag_axis = env.diag_axis;
        const bool diag_try_transpose = env.diag_try_transpose;
        const bool compare_replicated = env.compare_replicated;
        // Optional: small-matrix structural dump for debugging distributed layout vs row-major.
        // Enabled when LLAMINAR_COSMA_DUMP_SMALL is set and dimensions within limit (<=8 each).
        auto dump_small = [&](const char *tag, const CosmaView &V, int rows, int cols)
        {
            int local_flag = dump_small_enabled ? 1 : 0;
            int all_flag = local_flag;
            if (mpi_is_initialized() && mpi_world_size_safe() > 1)
            {
                MPI_Allreduce(&local_flag, &all_flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            }
            if (!all_flag)
                return;
            int limit = 8; // hard cap
            if (rows > limit || cols > limit)
                return;
            // Reconstruct (normalize=true for A/B, false for outputs decided by caller)
            std::vector<float> recon((size_t)rows * cols, 0.f);
            reconstruct_matrix_impl(V, recon.data(), V.label != 'C', env);
            if (rank_ == 0 && should_log(4))
            {
                LOG_TRACE("[CosmaPrefill][dump] " << tag << " global=" << rows << "x" << cols);
            }
            for (int gi = 0; gi < rows; ++gi)
            {
                for (int gj = 0; gj < cols; ++gj)
                {
                    int owner_rank = -1;
                    int local_index = -1;
                    if (V.mat)
                    {
                        auto lc = V.mat->local_coordinates(gi, gj);
                        local_index = lc.first;
                        owner_rank = lc.second;
                    }
                    if (rank_ == 0)
                    {
                        LOG_TRACE("[CosmaPrefill][dump] " << tag << " (" << gi << "," << gj << ") owner=" << owner_rank << " lidx=" << local_index << " val=" << recon[(size_t)gi * cols + gj]);
                    }
                }
            }
        };

        dump_small("A", A, m, k);
        dump_small("B", W.view, k, n);
        if (diag_enabled)
        {
            // Pre-matmul diagnostics: checksum A and B vs originals
            diag_global_checksum_impl(A, "A", env);
            diag_global_checksum_impl(W.view, "B", env);
            if (A.original_row_major)
                diag_sample_points_impl(A, A.original_row_major, "A", env);
            if (W.view.original_row_major)
                diag_sample_points_impl(W.view, W.view.original_row_major, "B", env);
            // TAP (env: LLAMINAR_COSMA_DIAG_TAP=<N>) dumps top-left NxN submatrices of A and B after reconstruction
            if (diag_tap_enabled)
            {
                int tap_m = std::min(m, diag_tap_value);
                int tap_k = std::min(k, diag_tap_value);
                int tap_n = std::min(n, diag_tap_value);
                std::vector<float> A_recon((size_t)m * k, 0.f);
                std::vector<float> B_recon((size_t)k * n, 0.f);
                reconstruct_matrix_impl(A, A_recon.data(), true, env);
                reconstruct_matrix_impl(W.view, B_recon.data(), true, env);
                if (rank_ == 0 && should_log(4))
                {
                    LOG_TRACE("[CosmaPrefill][Tap] A top-left " << tap_m << "x" << tap_k);
                    for (int i = 0; i < tap_m; ++i)
                    {
                        std::ostringstream row;
                        row << "A[" << i << "]:";
                        for (int j = 0; j < tap_k; ++j)
                        {
                            row << ' ' << A_recon[(size_t)i * k + j];
                        }
                        LOG_TRACE(row.str());
                    }
                    LOG_TRACE("[CosmaPrefill][Tap] B top-left " << tap_k << "x" << tap_n);
                    for (int i = 0; i < tap_k; ++i)
                    {
                        std::ostringstream row;
                        row << "B[" << i << "]:";
                        for (int j = 0; j < tap_n; ++j)
                        {
                            row << ' ' << B_recon[(size_t)i * n + j];
                        }
                        LOG_TRACE(row.str());
                    }
                }
            }
        }
        if (world_size_ == 1)
        {
            LOG_INFO("[CosmaPrefill] Using single-rank cblas fallback");
            stats_.single_rank_calls++;
            CosmaView C;
            C.global_rows = m;
            C.global_cols = n;
            C.label = 'C';
            C.host_owned = std::make_shared<std::vector<float>>(m * n, 0.f);
            const float *A_ptr = A.original_row_major;
            const float *B_ptr = W.view.original_row_major;
            if (!A_ptr || !B_ptr)
            {
                LOG_ERROR("[CosmaPrefill] Missing original row-major pointers for single-rank fallback");
                return C;
            }
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        transposeW ? CblasTrans : CblasNoTrans,
                        m, n, k,
                        alpha,
                        A_ptr, k,
                        B_ptr, n,
                        beta,
                        C.host_owned->data(), n);
            return C;
        }

        // Optional replicated fallback for multi-rank correctness debugging.
        // Enable by exporting LLAMINAR_COSMA_FORCE_REPLICATED_DIAG=1
        bool force_replicated = force_replicated_diag;
        if (force_replicated)
        {
            if (rank_ == 0)
                LOG_WARN("[CosmaPrefill][debug-force-fallback] Forcing replicated path (env override)");
            CosmaView C_rep;
            C_rep.global_rows = m;
            C_rep.global_cols = n;
            C_rep.label = 'C';
            C_rep.host_owned = std::make_shared<std::vector<float>>((size_t)m * n, 0.f);
            // Reconstruct full A and B (normalization true for inputs)
            std::vector<float> A_full((size_t)m * k, 0.f), B_full((size_t)k * n, 0.f);
            if (A.original_row_major)
            {
                std::memcpy(A_full.data(), A.original_row_major, (size_t)m * k * sizeof(float));
            }
            else
            {
                reconstruct_matrix_impl(A, A_full.data(), true, env);
            }
            if (W.view.original_row_major)
            {
                std::memcpy(B_full.data(), W.view.original_row_major, (size_t)k * n * sizeof(float));
            }
            else
            {
                reconstruct_matrix_impl(W.view, B_full.data(), true, env);
            }
            if (rank_ == 0)
            {
                double sumA = 0.0, sumB = 0.0;
                for (float v : A_full)
                    sumA += v;
                for (float v : B_full)
                    sumB += v;
                LOG_WARN("[CosmaPrefill][debug-force-fallback] sums A=" << sumA << " B=" << sumB);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, transposeW ? CblasTrans : CblasNoTrans,
                            m, n, k, alpha, A_full.data(), k, B_full.data(), n, beta, C_rep.host_owned->data(), n);
            }
            safe_bcast(C_rep.host_owned->data(), m * n, MPI_FLOAT, 0);
            if (validate_tile_tokens_ > 0)
                maybe_validation_tile_gemm(A_full.data(), B_full.data(), C_rep.host_owned->data(), m, k, n);
            stats_.fast_path_calls++;
            if (rank_ == 0 && should_log(2))
            {
                LOG_INFO("[CosmaPrefill][debug-force-fallback] fast_path_calls=" << stats_.fast_path_calls.load());
            }
            dump_small("C", C_rep, m, n);
            return C_rep;
        }
        // Multi-rank small-op fast path: replicate compute locally then Allreduce (sum) so every rank has same C
        long long volume = 1ll * m * n * k;
        if (world_size_ > 1 && volume < fast_path_threshold_ops_)
        {
            if (rank_ == 0)
            {
                LOG_INFO("[CosmaPrefill] Using multi-rank fast path (local BLAS + Allreduce) volume=" << volume);
            }
            stats_.fast_path_calls++;
            CosmaView C;
            C.global_rows = m;
            C.global_cols = n;
            C.label = 'C';
            C.host_owned = std::make_shared<std::vector<float>>(m * n, 0.f);
            const float *A_ptr = A.original_row_major; // In Phase 1 we assume replicated inputs/weights
            const float *B_ptr = W.view.original_row_major;
            if (!A_ptr || !B_ptr)
            {
                LOG_ERROR("[CosmaPrefill] Missing row-major pointers for fast path");
                return C;
            }
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        transposeW ? CblasTrans : CblasNoTrans,
                        m, n, k,
                        alpha,
                        A_ptr, k,
                        B_ptr, n,
                        beta,
                        C.host_owned->data(), n);
            // Replicated inputs => each rank already has identical result; broadcast rank 0's buffer for safety
            if (world_size_ > 1)
            {
                MPI_Bcast(C.host_owned->data(), m * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
            }
            return C;
        }
        // Use result strategy for C. NOTE: A and W may have been created with operand-specific strategies.
        auto &stratC = strategy_cache_.get(m, n, k, world_size_);
        // Temporary correctness safeguard: allow forcing replicated GEMM even for large ops
        // until full distributed layout population (proper block-cyclic mapping) is implemented.
        if (force_large_replicated)
        {
            if (rank_ == 0 && should_log(2))
            {
                LOG_INFO("[CosmaPrefill] FORCED replicated fallback for large matmul m=" << m << " n=" << n << " k=" << k);
            }
            stats_.fast_path_calls++; // reuse fast path counter to avoid adding new stat
            CosmaView C_rep;
            C_rep.global_rows = m;
            C_rep.global_cols = n;
            C_rep.label = 'C';
            C_rep.host_owned = std::make_shared<std::vector<float>>(m * n, 0.f);
            const float *A_ptr = A.original_row_major;
            const float *B_ptr = W.view.original_row_major;
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        transposeW ? CblasTrans : CblasNoTrans,
                        m, n, k,
                        alpha,
                        A_ptr, k,
                        B_ptr, n,
                        beta,
                        C_rep.host_owned->data(), n);
            MPI_Bcast(C_rep.host_owned->data(), m * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
            return C_rep;
        }
        // Decide whether to use experimental direct strategy path or C API layout path (default).
        // Allow completely disabling COSMA large path via existing adaptive override or new env.
        bool disable_cosma = cosma_disabled;
        // volume previously computed earlier (m*n*k) but may be out of scope if refactored; recompute safely
        long long full_volume = 1ll * m * n * k;
        bool use_direct = !disable_cosma && (force_direct || full_volume >= direct_threshold_ops);
        if (transposeW && use_direct)
        {
            use_direct = false;
            if (rank_ == 0 && should_log(1))
            {
                LOG_WARN("[CosmaPrefill] Disabling direct COSMA path due to transpose requirement");
            }
        }
        // Diagnostic replicated-B isolation (env LLAMINAR_COSMA_REPLICATE_B=1) keeps distributed A but broadcasts B content identically to all ranks.
        stats_.cosma_path_calls++;
        if (disable_cosma)
        {
            if (rank_ == 0 && should_log(1))
            {
                LOG_WARN("[CosmaPrefill] COSMA disabled via env; using replicated fallback");
            }
        }
        if (!use_direct) // fallback path
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (rank_ == 0 && should_log(2))
            {
                LOG_INFO("[CosmaPrefill][fallback-replicated] begin m=" << m << " n=" << n << " k=" << k
                                                                        << " alpha=" << alpha << " beta=" << beta);
            }
            if (should_log(3))
            {
                LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " entering replicated fallback");
            }
            CosmaView C_rep;
            C_rep.global_rows = m;
            C_rep.global_cols = n;
            C_rep.label = 'C';
            C_rep.host_owned = std::make_shared<std::vector<float>>((size_t)m * n, 0.f);

            std::vector<float> A_full_storage((size_t)m * k, 0.f);
            std::vector<float> B_full_storage((size_t)k * n, 0.f);

            auto broadcast_if_needed = [&](std::vector<float> &buffer, const float *source_ptr, size_t elements)
            {
                if (world_size_ <= 1)
                {
                    if (source_ptr && source_ptr != buffer.data())
                    {
                        std::memcpy(buffer.data(), source_ptr, elements * sizeof(float));
                    }
                    return;
                }
                if (rank_ == 0)
                {
                    if (source_ptr && source_ptr != buffer.data())
                    {
                        std::memcpy(buffer.data(), source_ptr, elements * sizeof(float));
                    }
                }
                safe_bcast(buffer.data(), static_cast<int>(elements), MPI_FLOAT, 0);
            };

            int use_A_broadcast_flag = (rank_ == 0 && A.original_row_major) ? 1 : 0;
            int use_B_broadcast_flag = (rank_ == 0 && W.view.original_row_major) ? 1 : 0;
            safe_bcast(&use_A_broadcast_flag, 1, MPI_INT, 0);
            safe_bcast(&use_B_broadcast_flag, 1, MPI_INT, 0);
            bool use_A_broadcast = (use_A_broadcast_flag != 0);
            bool use_B_broadcast = (use_B_broadcast_flag != 0);

            if (rank_ == 0 && should_log(2))
            {
                LOG_INFO("[CosmaPrefill][fallback-replicated] broadcast decisions A=" << use_A_broadcast
                                                                                      << " B=" << use_B_broadcast);
            }

            if (use_A_broadcast)
            {
                const float *src_ptr = (rank_ == 0) ? A.original_row_major : nullptr;
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " preparing A via broadcast");
                }
                broadcast_if_needed(A_full_storage, src_ptr, (size_t)m * k);
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " completed A broadcast");
                }
            }
            else
            {
                if (rank_ == 0 && should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] reconstructing A (missing shared row-major pointer)");
                }
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " reconstructing A");
                }
                reconstruct_matrix_impl(A, A_full_storage.data(), true, env);
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " finished reconstructing A");
                }
            }

            if (use_B_broadcast)
            {
                const float *src_ptr = (rank_ == 0) ? W.view.original_row_major : nullptr;
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " preparing B via broadcast");
                }
                broadcast_if_needed(B_full_storage, src_ptr, (size_t)k * n);
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " completed B broadcast");
                }
            }
            else
            {
                if (rank_ == 0 && should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] reconstructing B (missing shared row-major pointer)");
                }
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " reconstructing B");
                }
                reconstruct_matrix_impl(W.view, B_full_storage.data(), true, env);
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " finished reconstructing B");
                }
            }

            if (rank_ == 0)
            {
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] executing cblas_sgemm");
                }
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            transposeW ? CblasTrans : CblasNoTrans,
                            m, n, k,
                            alpha,
                            A_full_storage.data(), k,
                            B_full_storage.data(), n,
                            beta,
                            C_rep.host_owned->data(), n);
            }

            if (mpi_is_initialized() && mpi_world_size_safe() > 1)
            {
                if (rank_ == 0 && should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] broadcasting result");
                }
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " entering result broadcast");
                }
                safe_bcast(C_rep.host_owned->data(), m * n, MPI_FLOAT, 0);
                if (should_log(3))
                {
                    LOG_DEBUG("[CosmaPrefill][fallback-replicated] rank=" << rank_ << " completed result broadcast");
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            stats_.us_matmul += (long long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            stats_.matmul_invocations++;

            const float *A_validation = (rank_ == 0) ? (A_full_storage.empty() ? nullptr : A_full_storage.data()) : nullptr;
            const float *B_validation = (rank_ == 0) ? (B_full_storage.empty() ? nullptr : B_full_storage.data()) : nullptr;
            if (validate_tile_tokens_ > 0)
            {
                maybe_validation_tile_gemm(A_validation, B_validation, C_rep.host_owned->data(), m, k, n);
            }
            if (rank_ == 0 && should_log(2))
            {
                LOG_INFO("[CosmaPrefill][fallback-replicated] matmul m=" << m << " n=" << n << " k=" << k);
            }
            return C_rep;
        }
        // Experimental direct path (existing code) retained below
        CosmaView C;
        // If requested, reconstruct full B and redistribute identical values into each rank's local portion to isolate mapping issues.
        if (replicate_B && W.view.mat)
        {
            std::vector<float> B_full((size_t)k * n, 0.f);
            if (W.view.original_row_major)
            {
                std::memcpy(B_full.data(), W.view.original_row_major, (size_t)k * n * sizeof(float));
            }
            else
            {
                reconstruct_matrix_impl(W.view, B_full.data(), true, env);
            }
            float *b_local = W.view.mat->matrix_pointer();
            size_t b_sz = W.view.mat->matrix_size();
            if (b_local && b_sz)
            {
                std::fill(b_local, b_local + b_sz, 0.f);
                for (int gi = 0; gi < k; ++gi)
                {
                    for (int gj = 0; gj < n; ++gj)
                    {
                        auto lc = W.view.mat->local_coordinates(gi, gj);
                        if (lc.second == rank_)
                        {
                            int lidx = lc.first;
                            if (lidx >= 0 && (size_t)lidx < b_sz)
                                b_local[lidx] = B_full[(size_t)gi * n + gj];
                        }
                    }
                }
            }
            if (rank_ == 0 && should_log(2))
            {
                LOG_INFO("[CosmaPrefill][ReplicateB] Applied replicated-B diagnostic path");
            }
        }
        if (transposeW && rank_ == 0 && should_log(1))
        {
            LOG_WARN("[CosmaPrefill] transposeW=true path not implemented; forcing replicated fallback");
        }
        if (rank_ == 0 && should_log(2))
        {
            LOG_INFO("[CosmaPrefill][direct] matmul m=" << m << " n=" << n << " k=" << k
                                                        << " alpha=" << alpha << " beta=" << beta << " world_size=" << world_size_
                                                        << " volume=" << full_volume);
        }
        // Strategy selection: COSMA permits different strategies per operand (A: m x k, B: k x n, C: m x n).
        // Earlier we forced a unified strategy (stratC) for A and B, which appears to produce a systematic
        // permutation mismatch (relA≈1, cos≈0.5). Here we revert to per-operand strategies unless explicitly
        // overridden by an env flag (LLAMINAR_COSMA_FORCE_UNIFIED) for regression diagnostics.
        const auto &sA = strategy_cache_.get(m, k, k, world_size_);
        const auto &sB = strategy_cache_.get(k, n, k, world_size_);
        const auto &sC = strategy_cache_.get(m, n, k, world_size_);
        const bool use_unified_strategy = force_unified_strategy || force_direct;
        const cosma::Strategy &chosenA = use_unified_strategy ? sC : sA;
        const cosma::Strategy &chosenB = use_unified_strategy ? sC : sB;
        const cosma::Strategy &stratC_ref = sC; // result always uses (m,n,k)

        CosmaView A_compat = A; // start with original view
        CosmaView B_compat = W.view;
        bool rebuild_A = world_size_ > 1; // distributed path requires proper layout
        bool rebuild_B = world_size_ > 1;
        std::vector<float> A_full_buf, B_full_buf;
        if (rebuild_A)
        {
            if (rank_ == 0 && should_log(3))
                LOG_DEBUG("[CosmaPrefill][direct] Rebuilding A with unified strategy");
            // Reconstruct full A row-major
            A_full_buf.resize((size_t)m * k, 0.f);
            if (A.original_row_major)
            {
                std::memcpy(A_full_buf.data(), A.original_row_major, (size_t)m * k * sizeof(float));
            }
            else
            {
                reconstruct_matrix_impl(A, A_full_buf.data(), true, env);
            }
            A_compat = allocate_matrix('A', m, k, chosenA, false);
            fill_activation(A_compat, A_full_buf.data(), m, k, env);
            A_compat.original_row_major = A_full_buf.data();
        }
        if (rebuild_B)
        {
            if (rank_ == 0 && should_log(3))
                LOG_DEBUG("[CosmaPrefill][direct] Rebuilding B with unified strategy");
            B_full_buf.resize((size_t)k * n, 0.f);
            if (W.view.original_row_major)
            {
                std::memcpy(B_full_buf.data(), W.view.original_row_major, (size_t)k * n * sizeof(float));
            }
            else
            {
                reconstruct_matrix_impl(W.view, B_full_buf.data(), true, env);
            }
            B_compat = allocate_matrix('B', k, n, chosenB, false);
            // Scatter B_full_buf into B_compat just like stream_weight_blocks
            if (B_compat.mat && B_compat.mat->matrix_pointer())
            {
                float *local = B_compat.mat->matrix_pointer();
                size_t sz = B_compat.mat->matrix_size();
                std::fill(local, local + sz, 0.f);
                bool legacy_forward = pop_forward_legacy;
                bool dest_local_mode = !legacy_forward;
                bool trace_scatter_B = diag_deep && should_log(4);
                if (dest_local_mode)
                {
                    if (trace_scatter_B && rank_ == 0)
                        LOG_TRACE("[CosmaPrefill][pop] destination-local population (B) default");
                    for (size_t li = 0; li < sz; ++li)
                    {
                        auto gc = B_compat.mat->global_coordinates((int)li);
                        int gi = gc.first;
                        int gj = gc.second; // B dims k x n
                        if (gi >= 0 && gi < k && gj >= 0 && gj < n)
                        {
                            local[li] = B_full_buf[(size_t)gi * n + gj];
                            if (diag_deep && gi < std::min(k, 8) && gj < std::min(n, 8) && rank_ == 0 && should_log(4))
                            {
                                LOG_TRACE("[CosmaPrefill][scatterB-dest] li=" << li << " (gi=" << gi << ",gj=" << gj << ") val=" << local[li]);
                            }
                        }
                    }
                }
                else
                {
                    for (int gi = 0; gi < k; ++gi)
                    {
                        const float *row_ptr = B_full_buf.data() + (size_t)gi * n;
                        for (int gj = 0; gj < n; ++gj)
                        {
                            auto lc = B_compat.mat->local_coordinates(gi, gj);
                            if (lc.second == rank_)
                            {
                                int lidx = lc.first;
                                if (lidx >= 0 && (size_t)lidx < sz)
                                    local[lidx] = row_ptr[gj];
                                if (diag_deep && gi < std::min(k, 8) && gj < std::min(n, 8) && rank_ == 0 && should_log(4))
                                {
                                    LOG_TRACE("[CosmaPrefill][scatterB] (gi=" << gi << ",gj=" << gj << ") val=" << row_ptr[gj] << " lidx=" << lidx);
                                }
                            }
                        }
                    }
                }
                if (diag_local_probe && rank_ == 0)
                {
                    size_t szB = B_compat.mat->matrix_size();
                    size_t sampleB = std::min(szB, (size_t)4096);
                    long double numB = 0.0L, denB = 0.0L;
                    size_t mismatchesB = 0;
                    int loggedB = 0;
                    for (size_t li = 0; li < sampleB; ++li)
                    {
                        auto gc = B_compat.mat->global_coordinates((int)li);
                        int gi2 = gc.first, gj2 = gc.second;
                        if (gi2 < 0 || gj2 < 0 || gi2 >= k || gj2 >= n)
                            continue;
                        float v = local[li];
                        float ref = B_full_buf[(size_t)gi2 * n + gj2];
                        long double diff = (long double)v - (long double)ref;
                        numB += diff * diff;
                        denB += (long double)ref * ref;
                        if (v != ref && loggedB < 8)
                        {
                            LOG_INFO("[CosmaPrefill][local-probe] B mismatch li=" << li << " -> (gi=" << gi2 << ",gj=" << gj2 << ") v=" << v << " ref=" << ref);
                            ++loggedB;
                            ++mismatchesB;
                        }
                        else if (v != ref)
                            ++mismatchesB;
                    }
                    double relBpref = (denB > 0) ? (double)(std::sqrt(numB) / (std::sqrt(denB) + 1e-30L)) : 0.0;
                    LOG_INFO("[CosmaPrefill][local-probe] B prefix_rel_l2=" << relBpref << " sample=" << sampleB << " mismatches=" << mismatchesB);
                }
            }
            B_compat.original_row_major = B_full_buf.data();
            // Deep random-sampling local probe across the full local buffer to find first mismatch region.
            if (diag_local_probe_deep)
            {
                auto do_deep_probe = [&](const char *tag, const CosmaView &V, const float *ref, int rows, int cols)
                {
                    if (!V.mat || !ref)
                        return;
                    float *local = V.mat->matrix_pointer();
                    size_t sz = V.mat->matrix_size();
                    if (!local || !sz)
                        return;
                    const int SAMPLES = 128;
                    size_t seed = (size_t)(12345 + rank_);
                    auto next_rand = [&]()
                    {
                        // xorshift64*
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        return seed;
                    };
                    long double num = 0.0L, den = 0.0L;
                    int mismatches = 0;
                    size_t first_mismatch_li = (size_t)-1;
                    int first_gi = -1, first_gj = -1;
                    float first_v = 0.f, first_ref = 0.f;
                    std::vector<size_t> perm_signature;
                    perm_signature.reserve(SAMPLES);
                    for (int s = 0; s < SAMPLES; ++s)
                    {
                        size_t li = next_rand() % sz;
                        auto gc = V.mat->global_coordinates((int)li);
                        int gi = gc.first, gj = gc.second;
                        if (gi < 0 || gj < 0 || gi >= rows || gj >= cols)
                            continue;
                        float v = local[li];
                        float r = ref[(size_t)gi * cols + gj];
                        long double d = (long double)v - (long double)r;
                        num += d * d;
                        den += (long double)r * r;
                        if (v != r)
                        {
                            ++mismatches;
                            if (first_mismatch_li == (size_t)-1)
                            {
                                first_mismatch_li = li;
                                first_gi = gi;
                                first_gj = gj;
                                first_v = v;
                                first_ref = r;
                            }
                        }
                        perm_signature.push_back(((size_t)gi * cols + gj));
                    }
                    double rel = (den > 0) ? (double)(std::sqrt(num) / (std::sqrt(den) + 1e-30L)) : 0.0;
                    if (rank_ == 0 && should_log(2))
                    {
                        LOG_INFO(std::string("[CosmaPrefill][local-probe-deep] ") + tag + " samples=" + std::to_string(SAMPLES) + " rel_l2=" + std::to_string(rel) + " mismatches=" + std::to_string(mismatches));
                        if (first_mismatch_li != (size_t)-1)
                        {
                            LOG_INFO("[CosmaPrefill][local-probe-deep] first_mismatch tag=" << tag << " li=" << first_mismatch_li << " (gi=" << first_gi << ",gj=" << first_gj << ") v=" << first_v << " ref=" << first_ref);
                        }
                        // Emit a compact permutation hash: sort and diff between consecutive global linear indices
                        if (!perm_signature.empty())
                        {
                            std::vector<size_t> tmp = perm_signature;
                            std::sort(tmp.begin(), tmp.end());
                            size_t gaps = 0;
                            for (size_t i = 1; i < tmp.size(); ++i)
                                if (tmp[i] != tmp[i - 1] + 1)
                                    ++gaps;
                            LOG_INFO("[CosmaPrefill][local-probe-deep] tag=" << tag << " perm_gaps=" << gaps);
                        }
                    }
                };
                do_deep_probe("A", A_compat, A_full_buf.data(), m, k);
                do_deep_probe("B", B_compat, B_full_buf.data(), k, n);
            }
        }
        // Deep population diagnostics (gated) to determine mismatch cause before multiply
        if (!C.mat)
        {
            C = allocate_matrix('C', m, n, stratC, beta == 0.f);
        }
        bool deep_diag = diag_deep && (world_size_ > 1);
        std::vector<float> A_pop, B_pop, C_pop_ref, C_orig_ref; // buffers reused across diag
        if (deep_diag)
        {
            double t_pop_start = MPI_Wtime();
            A_pop.resize((size_t)m * k);
            B_pop.resize((size_t)k * n);
            bool bypass = diag_recon_bypass;
            bool transpose_exp = diag_recon_transpose;
            bool swap_rc_exp = diag_swaprc;
            if (bypass)
            {
                // Directly copy original references (rank0 owns canonical) then broadcast so all ranks share identical buffer.
                const float *A_ref_raw = A_compat.original_row_major ? A_compat.original_row_major : A.original_row_major;
                const float *B_ref_raw = B_compat.original_row_major ? B_compat.original_row_major : W.view.original_row_major;
                if (A_ref_raw)
                    std::memcpy(A_pop.data(), A_ref_raw, (size_t)m * k * sizeof(float));
                else
                    std::fill(A_pop.begin(), A_pop.end(), 0.f);
                if (B_ref_raw)
                    std::memcpy(B_pop.data(), B_ref_raw, (size_t)k * n * sizeof(float));
                else
                    std::fill(B_pop.begin(), B_pop.end(), 0.f);
                if (mpi_is_initialized() && mpi_world_size_safe() > 1)
                {
                    MPI_Bcast(A_pop.data(), (int)A_pop.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(B_pop.data(), (int)B_pop.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
                }
                if (rank_ == 0 && should_log(3))
                    LOG_INFO("[CosmaPrefill][diag] Reconstruction BYPASS active");
            }
            else
            {
                reconstruct_matrix_impl(A_compat, A_pop.data(), true, env);
                reconstruct_matrix_impl(B_compat, B_pop.data(), true, env);
            }
            // Optional alternate reconstruction treating returned global_coordinates() pair as (col,row) instead of (row,col)
            std::vector<float> A_pop_swap, B_pop_swap;
            double relA_swap = -1.0, relB_swap = -1.0;
            if (!bypass && swap_rc_exp)
            {
                if (rank_ == 0 && should_log(2))
                    LOG_INFO("[CosmaPrefill][swaprc] Alternate swap(row,col) reconstruction enabled");
                if (A_compat.mat && B_compat.mat)
                {
                    A_pop_swap.assign((size_t)m * k, 0.f);
                    B_pop_swap.assign((size_t)k * n, 0.f);
                    // Reconstruct A assuming gc.first=col (0..k-1), gc.second=row (0..m-1)
                    auto do_swap_recon = [&](const CosmaView &view, std::vector<float> &out, int rows, int cols)
                    {
                        if (!view.mat)
                            return;
                        float *local = view.mat->matrix_pointer();
                        size_t lsz = view.mat->matrix_size();
                        // Gather local triplets
                        std::vector<int> gi_send(lsz), gj_send(lsz);
                        std::vector<float> val_send(lsz);
                        for (size_t li = 0; li < lsz; ++li)
                        {
                            auto gc = view.mat->global_coordinates((int)li);
                            gi_send[li] = gc.first;
                            gj_send[li] = gc.second;
                            val_send[li] = local[li];
                        }
                        // Root gather counts
                        int local_n = (int)lsz;
                        std::vector<int> counts(world_size_, 0);
                        std::vector<int> displs(world_size_, 0);
                        MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                        int total_entries = 0;
                        if (rank_ == 0)
                        {
                            for (int r = 0; r < world_size_; ++r)
                            {
                                displs[r] = total_entries;
                                total_entries += counts[r];
                            }
                        }
                        std::vector<int> gi_all, gj_all;
                        std::vector<float> val_all;
                        if (rank_ == 0)
                        {
                            gi_all.resize(total_entries);
                            gj_all.resize(total_entries);
                            val_all.resize(total_entries);
                        }
                        MPI_Gatherv(gi_send.data(), local_n, MPI_INT, rank_ == 0 ? gi_all.data() : nullptr, rank_ == 0 ? counts.data() : nullptr, rank_ == 0 ? displs.data() : nullptr, MPI_INT, 0, MPI_COMM_WORLD);
                        MPI_Gatherv(gj_send.data(), local_n, MPI_INT, rank_ == 0 ? gj_all.data() : nullptr, rank_ == 0 ? counts.data() : nullptr, rank_ == 0 ? displs.data() : nullptr, MPI_INT, 0, MPI_COMM_WORLD);
                        MPI_Gatherv(val_send.data(), local_n, MPI_FLOAT, rank_ == 0 ? val_all.data() : nullptr, rank_ == 0 ? counts.data() : nullptr, rank_ == 0 ? displs.data() : nullptr, MPI_FLOAT, 0, MPI_COMM_WORLD);
                        if (rank_ == 0)
                        {
                            std::fill(out.begin(), out.end(), 0.f);
                            for (int i = 0; i < total_entries; ++i)
                            {
                                int gc0 = gi_all[i];
                                int gc1 = gj_all[i]; // interpret swapped
                                if (gc1 >= 0 && gc1 < rows && gc0 >= 0 && gc0 < cols)
                                {
                                    size_t w = (size_t)gc1 * cols + gc0;
                                    out[w] = val_all[i];
                                }
                            }
                        }
                        MPI_Bcast(out.data(), (int)out.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
                    };
                    do_swap_recon(A_compat, A_pop_swap, m, k);
                    do_swap_recon(B_compat, B_pop_swap, k, n);
                }
            }
            // Select the most accurate reference pointers: prefer compat originals captured during population
            const float *A_ref = A_compat.original_row_major ? A_compat.original_row_major : A.original_row_major;
            const float *B_ref = B_compat.original_row_major ? B_compat.original_row_major : W.view.original_row_major;
            auto rel_l2 = [&](const std::vector<float> &X, const float *Y, size_t sz)
            {
                if (!Y) return -1.0; long double num=0.0L, den=0.0L; for(size_t i=0;i<sz;++i){ long double d=(long double)X[i]-(long double)Y[i]; num+=d*d; long double r=(long double)Y[i]; den+=r*r; } return (double)(std::sqrt(num)/(std::sqrt(den)+1e-30L)); };
            double relA = rel_l2(A_pop, A_ref, (size_t)m * k);
            double relB = rel_l2(B_pop, B_ref, (size_t)k * n);
            if (!A_pop_swap.empty())
                relA_swap = rel_l2(A_pop_swap, A_ref, (size_t)m * k);
            if (!B_pop_swap.empty())
                relB_swap = rel_l2(B_pop_swap, B_ref, (size_t)k * n);
            double relA_trans = -1.0, relB_trans = -1.0;
            if (!bypass && transpose_exp)
            {
                // Reconstruct again using transpose indexing to test hypothesis of swapped dims
                std::vector<float> A_pop_T((size_t)m * k, 0.f), B_pop_T((size_t)k * n, 0.f);
                // Force reconstruct with transpose flag by calling reconstruct_matrix with env already set.
                reconstruct_matrix_impl(A_compat, A_pop_T.data(), true, env); // this will obey recon_transpose flag externally set
                reconstruct_matrix_impl(B_compat, B_pop_T.data(), true, env);
                relA_trans = rel_l2(A_pop_T, A_ref, (size_t)m * k);
                relB_trans = rel_l2(B_pop_T, B_ref, (size_t)k * n);
            }
            auto cosine = [](const std::vector<float> &X, const float *Y, size_t sz)
            { if(!Y) return -2.0; long double dot=0.0L, nx=0.0L, ny=0.0L; for(size_t i=0;i<sz;++i){ long double xv=X[i]; long double yv=Y[i]; dot+=xv*yv; nx+=xv*xv; ny+=yv*yv;} if(nx==0||ny==0) return -1.0; return (double)(dot / (std::sqrt(nx)*std::sqrt(ny))); };
            double cosA = cosine(A_pop, A_ref, (size_t)m * k);
            double cosB = cosine(B_pop, B_ref, (size_t)k * n);
            auto mean_abs_diff = [](const std::vector<float> &X, const float *Y, size_t sz)
            { if(!Y) return -1.0; long double acc=0.0L; for(size_t i=0;i<sz;++i) acc+= std::fabs((long double)X[i] - (long double)Y[i]); return (double)(acc / (long double)sz); };
            double madA = mean_abs_diff(A_pop, A_ref, (size_t)m * k);
            double madB = mean_abs_diff(B_pop, B_ref, (size_t)k * n);
            auto zero_frac = [](const std::vector<float> &X)
            { size_t z=0; for(float v: X) if (v==0.f) ++z; return (double)z/(double)X.size(); };
            double zA = zero_frac(A_pop);
            double zB = zero_frac(B_pop);
            // Permutation diagnostics: test even-odd row reorder hypothesis (rows even first then odd)
            auto rel_perm_rows = [&](const std::vector<float> &X, const float *Y, int rows, int cols)
            {
                if(!Y) return -1.0; std::vector<float> perm; perm.resize((size_t)rows*cols);
                int half = rows/2; int ei=0, oi=0; for(int r=0;r<rows;++r){ if((r & 1)==0){ int dst = ei++; for(int c=0;c<cols;++c) perm[(size_t)dst*cols + c] = Y[(size_t)r*cols + c]; } }
                for(int r=0;r<rows;++r){ if((r & 1)==1){ int dst = half + oi++; for(int c=0;c<cols;++c) perm[(size_t)dst*cols + c] = Y[(size_t)r*cols + c]; } }
                // compute rel L2 between X and perm
                long double num=0.0L, den=0.0L; size_t sz=(size_t)rows*cols; for(size_t i=0;i<sz;++i){ long double d=(long double)X[i] - (long double)perm[i]; num+=d*d; long double r=(long double)perm[i]; den+=r*r; }
                return (double)(std::sqrt(num)/(std::sqrt(den)+1e-30L)); };
            double relA_perm = rel_perm_rows(A_pop, A_ref, m, k);
            double relB_perm = rel_perm_rows(B_pop, B_ref, k, n);
            // Row & column sums first few for orientation clues
            int inspect_rows = std::min(m, 8);
            int inspect_colsA = std::min(k, 8);
            int inspect_colsB = std::min(n, 8);
            std::ostringstream rsA, csA, rsB, csB;
            rsA << "rsA:";
            csA << "csA:";
            rsB << "rsB:";
            csB << "csB:";
            for (int i = 0; i < inspect_rows; ++i)
            {
                double s = 0.0;
                for (int j = 0; j < k; ++j)
                    s += A_pop[(size_t)i * k + j];
                rsA << ' ' << s;
            }
            for (int j = 0; j < inspect_colsA; ++j)
            {
                double s = 0.0;
                for (int i = 0; i < m; ++i)
                    s += A_pop[(size_t)i * k + j];
                csA << ' ' << s;
            }
            for (int i = 0; i < std::min(k, 8); ++i)
            {
                double s = 0.0;
                for (int j = 0; j < n; ++j)
                    s += B_pop[(size_t)i * n + j];
                rsB << ' ' << s;
            }
            for (int j = 0; j < inspect_colsB; ++j)
            {
                double s = 0.0;
                for (int i = 0; i < k; ++i)
                    s += B_pop[(size_t)i * n + j];
                csB << ' ' << s;
            }
            // Optional permutation inference: attempt to prove A_pop (and similarly B_pop along k dimension) is a pure row permutation of reference.
            const char *env_perm_flag = diag_perm_spec;
            if (env_perm_flag && rank_ == 0)
            {
                std::ostringstream oss;
                oss << "[CosmaPrefill][perm-infer] env flag set value='" << env_perm_flag << "' relA=" << relA << " relB=" << relB;
                LOG_INFO(oss.str());
            }
            bool infer_perm = (env_perm_flag != nullptr); // always attempt if flag set
            if (rank_ == 0)
            {
                std::ostringstream oss;
                oss << "[CosmaPrefill][perm-infer] infer_perm decision=" << (infer_perm ? "true" : "false")
                    << " relA=" << relA << " relB=" << relB;
                if (!infer_perm)
                    oss << " (env flag absent)";
                else if (relA <= 0.5 && relB <= 0.5)
                    oss << " (proceeding even though rel values small)";
                LOG_INFO(oss.str());
            }
            if (infer_perm)
            {
                auto infer_row_permutation = [&](const std::vector<float> &pop, const float *ref, int rows, int cols, std::vector<int> &perm_out, double &rel_after)
                {
                    perm_out.clear();
                    rel_after = -1.0;
                    if (!ref)
                        return false;
                    perm_out.resize(rows, -1);
                    // Build 64-bit hash per reference row
                    struct RowHashEntry
                    {
                        uint64_t h;
                        int idx;
                    };
                    std::vector<RowHashEntry> ref_hash(rows);
                    auto row_hash = [&](const float *base)
                    {
                        // Simple mixing hash over row contents (depending on cols). Deterministic.
                        uint64_t h = 1469598103934665603ull; // FNV offset basis
                        const uint32_t *data = reinterpret_cast<const uint32_t *>(base);
                        for (int c = 0; c < cols; ++c)
                        {
                            uint64_t v = (uint64_t)data[c];
                            h ^= v;
                            h *= 1099511628211ull;
                        }
                        return h;
                    };
                    for (int r = 0; r < rows; ++r)
                    {
                        ref_hash[r] = {row_hash(ref + (size_t)r * cols), r};
                    }
                    // Sort for binary search (avoid unordered_map overhead for large 4k rows)
                    std::sort(ref_hash.begin(), ref_hash.end(), [](const RowHashEntry &a, const RowHashEntry &b)
                              { return a.h < b.h; });
                    auto find_ref = [&](uint64_t h) -> int
                    {
                        int lo=0, hi=rows-1; while(lo<=hi){ int mid=(lo+hi)/2; if(ref_hash[mid].h==h) return ref_hash[mid].idx; if(ref_hash[mid].h < h) lo=mid+1; else hi=mid-1; } return -1; };
                    std::vector<char> used(rows, 0);
                    bool ok = true;
                    for (int r = 0; r < rows; ++r)
                    {
                        uint64_t h = row_hash(pop.data() + (size_t)r * cols);
                        int ref_r = find_ref(h);
                        if (ref_r < 0 || used[ref_r])
                        {
                            ok = false;
                            break;
                        }
                        perm_out[r] = ref_r;
                        used[ref_r] = 1;
                    }
                    if (!ok)
                    {
                        return false;
                    }
                    // Apply inverse permutation to test alignment: we want reordered_pop[perm_out[r]] = pop[r]
                    std::vector<float> reordered;
                    reordered.resize((size_t)rows * cols);
                    for (int r = 0; r < rows; ++r)
                    {
                        int ref_r = perm_out[r];
                        if (ref_r >= 0)
                        {
                            std::memcpy(&reordered[(size_t)ref_r * cols], &pop[(size_t)r * cols], (size_t)cols * sizeof(float));
                        }
                    }
                    // Compute rel L2 between reordered and reference
                    long double num = 0.0L, den = 0.0L;
                    size_t sz = (size_t)rows * cols;
                    for (size_t i = 0; i < sz; ++i)
                    {
                        long double d = (long double)reordered[i] - (long double)ref[i];
                        num += d * d;
                        long double rv = (long double)ref[i];
                        den += rv * rv;
                    }
                    rel_after = (double)(std::sqrt(num) / (std::sqrt(den) + 1e-30L));
                    return true;
                };
                std::vector<int> permA;
                double relA_after = -1.0;
                bool have_permA = infer_row_permutation(A_pop, A_ref, m, k, permA, relA_after);
                if (have_permA && rank_ == 0)
                {
                    int mismatches = 0;
                    for (int i = 0; i < m; ++i)
                        if (permA[i] != i)
                            ++mismatches;
                    int first_print = 0;
                    std::ostringstream oss;
                    oss << "[CosmaPrefill][perm-infer] A row permutation size=" << m << " moved_rows=" << mismatches << " rel_after=" << relA_after;
                    LOG_INFO(oss.str());
                    std::ostringstream map_os;
                    map_os << "[CosmaPrefill][perm-infer] A perm sample:";
                    for (int i = 0; i < m && first_print < 16; ++i)
                    {
                        if (permA[i] != i)
                        {
                            map_os << " (" << i << "->" << permA[i] << ")";
                            ++first_print;
                        }
                    }
                    if (first_print > 0)
                        LOG_INFO(map_os.str());
                }
                else if (rank_ == 0)
                {
                    LOG_INFO("[CosmaPrefill][perm-infer] A permutation inference failed (either not pure permutation, hash collision, or ref null)");
                }
                // For B, permutation would be along the k dimension (rows of B_pop). Attempt only if relB large.
                std::vector<int> permB;
                double relB_after = -1.0;
                bool have_permB = false;
                have_permB = infer_row_permutation(B_pop, B_ref, k, n, permB, relB_after);
                if (have_permB && rank_ == 0)
                {
                    int mismatches = 0;
                    for (int i = 0; i < k; ++i)
                        if (permB[i] != i)
                            ++mismatches;
                    std::ostringstream oss;
                    oss << "[CosmaPrefill][perm-infer] B row permutation size=" << k << " moved_rows=" << mismatches << " rel_after=" << relB_after;
                    LOG_INFO(oss.str());
                }
                else if (rank_ == 0)
                {
                    LOG_INFO("[CosmaPrefill][perm-infer] B permutation inference failed (either not pure permutation, hash collision, or ref null)");
                }
            }
            // Axis diagnostics: attempt column permutation inference and transpose comparison if flag set.
            if (diag_axis)
            {
                if (rank_ == 0)
                    LOG_INFO("[CosmaPrefill][axis] Axis diagnostics enabled");
                auto infer_col_permutation = [&](const std::vector<float> &pop, const float *ref, int rows, int cols, std::vector<int> &perm_out, double &rel_after)
                {
                    perm_out.clear();
                    rel_after = -1.0;
                    if (!ref)
                        return false;
                    perm_out.resize(cols, -1);
                    struct ColHashEntry
                    {
                        uint64_t h;
                        int idx;
                    };
                    std::vector<ColHashEntry> ref_hash(cols);
                    auto col_hash = [&](int c)
                    { uint64_t h=1469598103934665603ull; for(int r=0;r<rows;++r){ uint64_t v = (uint64_t)reinterpret_cast<const uint32_t&>(ref[(size_t)r*cols + c]); h ^= v; h *= 1099511628211ull; } return h; };
                    for (int c = 0; c < cols; ++c)
                        ref_hash[c] = {col_hash(c), c};
                    std::sort(ref_hash.begin(), ref_hash.end(), [](auto &a, auto &b)
                              { return a.h < b.h; });
                    auto find_ref = [&](uint64_t h)
                    { int lo=0, hi=cols-1; while(lo<=hi){ int mid=(lo+hi)/2; if(ref_hash[mid].h==h) return ref_hash[mid].idx; if(ref_hash[mid].h < h) lo=mid+1; else hi=mid-1; } return -1; };
                    std::vector<char> used(cols, 0);
                    bool ok = true;
                    for (int c = 0; c < cols; ++c)
                    { // hash pop column c
                        uint64_t h = 1469598103934665603ull;
                        for (int r = 0; r < rows; ++r)
                        {
                            uint64_t v = (uint64_t)reinterpret_cast<const uint32_t &>(pop[(size_t)r * cols + c]);
                            h ^= v;
                            h *= 1099511628211ull;
                        }
                        int ref_c = find_ref(h);
                        if (ref_c < 0 || used[ref_c])
                        {
                            ok = false;
                            break;
                        }
                        perm_out[c] = ref_c;
                        used[ref_c] = 1;
                    }
                    if (!ok)
                        return false;
                    std::vector<float> reordered;
                    reordered.resize((size_t)rows * cols);
                    for (int c = 0; c < cols; ++c)
                    {
                        int dst = perm_out[c];
                        if (dst >= 0)
                        {
                            for (int r = 0; r < rows; ++r)
                                reordered[(size_t)r * cols + dst] = pop[(size_t)r * cols + c];
                        }
                    }
                    long double num = 0.0L, den = 0.0L;
                    size_t sz = (size_t)rows * cols;
                    for (size_t i = 0; i < sz; ++i)
                    {
                        long double d = (long double)reordered[i] - (long double)ref[i];
                        num += d * d;
                        long double rv = (long double)ref[i];
                        den += rv * rv;
                    }
                    rel_after = (double)(std::sqrt(num) / (std::sqrt(den) + 1e-30L));
                    return true;
                };
                // Transpose experiment (simple full transpose) for A: create A_T such that A_T[r*k + c] = A_pop[c*m + r] (treat original as transposed)
                auto rel_after_transpose = [&](const std::vector<float> &pop, const float *ref, int rows, int cols)
                { if(!ref) return -1.0; std::vector<float> T((size_t)rows*cols); for(int r=0;r<rows;++r) for(int c=0;c<cols;++c) T[(size_t)r*cols + c] = pop[(size_t)c*rows + r]; return rel_l2(T, ref, (size_t)rows*cols); };
                // A: column permutation
                std::vector<int> permA_cols;
                double relA_col_after = -1.0;
                bool have_colA = infer_col_permutation(A_pop, A_ref, m, k, permA_cols, relA_col_after);
                // B: column permutation (B dims k x n)
                std::vector<int> permB_cols;
                double relB_col_after = -1.0;
                bool have_colB = infer_col_permutation(B_pop, B_ref, k, n, permB_cols, relB_col_after);
                double relA_trans_simple = rel_after_transpose(A_pop, A_ref, m, k);
                double relB_trans_simple = rel_after_transpose(B_pop, B_ref, k, n);
                if (rank_ == 0)
                {
                    std::ostringstream oss;
                    oss << "[CosmaPrefill][axis] A col-perm=" << (have_colA ? "yes" : "no") << " rel_after=" << relA_col_after
                        << " simple_trans_rel=" << relA_trans_simple;
                    LOG_INFO(oss.str());
                    std::ostringstream oss2;
                    oss2 << "[CosmaPrefill][axis] B col-perm=" << (have_colB ? "yes" : "no") << " rel_after=" << relB_col_after
                         << " simple_trans_rel=" << relB_trans_simple;
                    LOG_INFO(oss2.str());
                    if (have_colA)
                    {
                        int printed = 0;
                        std::ostringstream map;
                        map << "[CosmaPrefill][axis] A col perm sample:";
                        for (int c = 0; c < k && printed < 16; ++c)
                        {
                            if (permA_cols[c] != c)
                            {
                                map << " (" << c << "->" << permA_cols[c] << ")";
                                ++printed;
                            }
                        }
                        if (printed)
                            LOG_INFO(map.str());
                    }
                    if (have_colB)
                    {
                        int printed = 0;
                        std::ostringstream map;
                        map << "[CosmaPrefill][axis] B col perm sample:";
                        for (int c = 0; c < n && printed < 16; ++c)
                        {
                            if (permB_cols[c] != c)
                            {
                                map << " (" << c << "->" << permB_cols[c] << ")";
                                ++printed;
                            }
                        }
                        if (printed)
                            LOG_INFO(map.str());
                    }
                }
            }
            // Ownership duplication check small tile (32x32 or bounds)
            int tile_m = std::min(m, 32), tile_k = std::min(k, 32), tile_n = std::min(n, 32);
            std::vector<int> ownA(tile_m * tile_k, 0), ownB(tile_k * tile_n, 0);
            if (A_compat.mat && A_compat.mat->matrix_pointer())
            {
                float *loc = A_compat.mat->matrix_pointer();
                size_t sz = A_compat.mat->matrix_size();
                for (size_t li = 0; li < sz; ++li)
                {
                    auto gc = A_compat.mat->global_coordinates((int)li);
                    if (gc.first < tile_m && gc.second < tile_k)
                        ownA[(size_t)gc.first * tile_k + gc.second] = 1;
                }
            }
            if (B_compat.mat && B_compat.mat->matrix_pointer())
            {
                float *loc = B_compat.mat->matrix_pointer();
                size_t sz = B_compat.mat->matrix_size();
                for (size_t li = 0; li < sz; ++li)
                {
                    auto gc = B_compat.mat->global_coordinates((int)li);
                    if (gc.first < tile_k && gc.second < tile_n)
                        ownB[(size_t)gc.first * tile_n + gc.second] = 1;
                }
            }
            if (mpi_is_initialized() && mpi_world_size_safe() > 1)
            {
                MPI_Allreduce(MPI_IN_PLACE, ownA.data(), (int)ownA.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, ownB.data(), (int)ownB.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
            int dupA = 0, dupB = 0;
            for (int v : ownA)
                if (v > 1)
                    ++dupA;
            for (int v : ownB)
                if (v > 1)
                    ++dupB;
            // Reference GEMMs
            C_pop_ref.resize((size_t)m * n, 0.f);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A_pop.data(), k, B_pop.data(), n, 0.f, C_pop_ref.data(), n);
            if (A.original_row_major && W.view.original_row_major)
            {
                C_orig_ref.resize((size_t)m * n, 0.f);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A.original_row_major, k, W.view.original_row_major, n, 0.f, C_orig_ref.data(), n);
            }
            double pre_rel = -1.0;
            if (!C_orig_ref.empty())
            {
                long double num = 0.0L, den = 0.0L;
                for (size_t i = 0; i < C_pop_ref.size(); ++i)
                {
                    long double d = (long double)C_pop_ref[i] - (long double)C_orig_ref[i];
                    num += d * d;
                    long double r = (long double)C_orig_ref[i];
                    den += r * r;
                }
                pre_rel = (double)(std::sqrt(num) / (std::sqrt(den) + 1e-30L));
            }
            // Orientation experiment (optional): if env LLAMINAR_COSMA_DIAG_TRY_TRANSPOSE is set, compute C using transposed B_pop
            double pre_rel_trans = -1.0; // rel_l2 between C_pop_ref (normal) and C computed with transposed B
            if (diag_try_transpose)
            {
                std::vector<float> B_pop_T((size_t)k * n, 0.f); // treat as n x k logically but store transposed for GEMM
                // Build transpose of B_pop (k x n -> n x k) then compute C_alt = A_pop * B_pop^T (meaning original orientation maybe wrong)
                for (int gi = 0; gi < k; ++gi)
                {
                    for (int gj = 0; gj < n; ++gj)
                    {
                        B_pop_T[(size_t)gj * k + gi] = B_pop[(size_t)gi * n + gj];
                    }
                }
                std::vector<float> C_alt((size_t)m * n, 0.f);
                // Compute with B transposed meaning we call GEMM with A (m x k) * B_T^T (k x n) => need B_T row-major as n x k used with Trans flag
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.f, A_pop.data(), k, B_pop_T.data(), k, 0.f, C_alt.data(), n);
                if (!C_orig_ref.empty())
                {
                    long double num = 0.0L, den = 0.0L;
                    for (size_t i = 0; i < C_alt.size(); ++i)
                    {
                        long double d = (long double)C_alt[i] - (long double)C_orig_ref[i];
                        num += d * d;
                        long double r = (long double)C_orig_ref[i];
                        den += r * r;
                    }
                    pre_rel_trans = (double)(std::sqrt(num) / (std::sqrt(den) + 1e-30L));
                }
            }
            double t_pop_end = MPI_Wtime();
            // Promote a concise summary to INFO for visibility even when TRACE not shown. Detailed vectors remain at TRACE.
            if (rank_ == 0 && should_log(2))
            {
                LOG_INFO("[CosmaPrefill][deep-summary] relA=" << relA << " relB=" << relB
                                                              << ((relA_swap >= 0) ? std::string(" relA_swap=") + std::to_string(relA_swap) : std::string(""))
                                                              << ((relB_swap >= 0) ? std::string(" relB_swap=") + std::to_string(relB_swap) : std::string(""))
                                                              << ((relA_trans >= 0) ? std::string(" relA_trans=") + std::to_string(relA_trans) : std::string(""))
                                                              << ((relB_trans >= 0) ? std::string(" relB_trans=") + std::to_string(relB_trans) : std::string(""))
                                                              << " cosA=" << cosA << " cosB=" << cosB << " madA=" << madA << " madB=" << madB << " zA=" << zA << " zB=" << zB
                                                              << " dupA=" << dupA << " dupB=" << dupB << " pre_rel_l2=" << pre_rel
                                                              << (pre_rel_trans >= 0 ? std::string(" pre_rel_l2_trans=") + std::to_string(pre_rel_trans) : std::string(""))
                                                              << " prep_ms=" << (t_pop_end - t_pop_start) * 1000.0);
                // Emit a concise mismatch sample if operands differ substantially from originals.
                auto emit_samples = [&](const char *tag, const std::vector<float> &pop, const float *orig, int R, int C)
                {
                    if(!orig) return; int printed=0; std::ostringstream oss; oss<<"[CosmaPrefill][deep-sample] "<<tag<<" mismatches:"; size_t totalRC=(size_t)R*C; for(size_t idx=0; idx<totalRC && printed<8; ++idx){ float pv=pop[idx]; float ov=orig[idx]; if(std::fabs(pv-ov) > 1e-5f){ int gi= idx / C; int gj = idx % C; oss<<" ("<<gi<<","<<gj<<":"<<ov<<"->"<<pv<<")"; ++printed; }} if(printed>0) LOG_INFO(oss.str()); };
                if (relA > 0.5)
                    emit_samples("A", A_pop, A_ref, m, k);
                if (relB > 0.5)
                    emit_samples("B", B_pop, B_ref, k, n);
                if (should_log(3))
                {
                    LOG_TRACE("[CosmaPrefill][deep] ref_pointers A_ref=" << (void *)A_ref << " (compat=" << (A_ref == A_compat.original_row_major) << ") B_ref=" << (void *)B_ref << " (compat=" << (B_ref == B_compat.original_row_major) << ")");
                }
            }
            if (rank_ == 0 && should_log(2))
            {
                LOG_INFO("[CosmaPrefill][deep-summary] relA=" << relA << " relB=" << relB
                                                              << ((relA_swap >= 0) ? std::string(" relA_swap=") + std::to_string(relA_swap) : std::string(""))
                                                              << ((relB_swap >= 0) ? std::string(" relB_swap=") + std::to_string(relB_swap) : std::string(""))
                                                              << ((relA_trans >= 0) ? std::string(" relA_trans=") + std::to_string(relA_trans) : std::string(""))
                                                              << ((relB_trans >= 0) ? std::string(" relB_trans=") + std::to_string(relB_trans) : std::string(""))
                                                              << " relA_perm=" << relA_perm << " relB_perm=" << relB_perm << " cosA=" << cosA << " cosB=" << cosB << " madA=" << madA << " madB=" << madB << " zA=" << zA << " zB=" << zB << " dupA=" << dupA << " dupB=" << dupB << " pre_rel_l2=" << pre_rel << " prep_ms=" << ((MPI_Wtime() - t_pop_start) * 1000.0));
                LOG_TRACE("[CosmaPrefill][deep] " << csA.str());
                LOG_TRACE("[CosmaPrefill][deep] " << rsB.str());
                LOG_TRACE("[CosmaPrefill][deep] " << csB.str());
            }
        }
        if (deep_diag && rank_ == 0 && should_log(4))
        {
            LOG_TRACE("[CosmaPrefill][direct] pre-multiply diagnostics done");
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        if (rank_ == 0 && should_log(2))
        {
            LOG_INFO("[CosmaPrefill][direct] entering pre-multiply barrier");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank_ == 0 && should_log(2))
        {
            LOG_INFO("[CosmaPrefill][direct] exited pre-multiply barrier");
        }
        if (rank_ == 0 && should_log(2))
        {
            LOG_INFO("[CosmaPrefill][direct] entering cosma::multiply");
        }
        try
        {
            cosma::multiply(*A_compat.mat, *B_compat.mat, *C.mat, stratC, MPI_COMM_WORLD, alpha, beta);
            if (rank_ == 0 && should_log(2))
            {
                LOG_INFO("[CosmaPrefill][direct] cosma::multiply completed");
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[CosmaPrefill][direct] cosma::multiply exception: " << e.what());
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank_ == 0 && should_log(2))
        {
            LOG_INFO("[CosmaPrefill][direct] exited post-multiply barrier");
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        stats_.us_matmul += (long long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        stats_.matmul_invocations++;
        if (deep_diag)
        {
            // Post multiply diagnostics: reconstruct C and compare
            std::vector<float> C_pop((size_t)m * n, 0.f);
            reconstruct_matrix_impl(C, C_pop.data(), false, env);
            if (rank_ == 0)
            {
                auto rel = [&](const std::vector<float> &R, const std::vector<float> &S)
                { if(R.empty()||S.empty()) return -1.0; long double num=0.0L, den=0.0L; for(size_t i=0;i<R.size();++i){ long double d=(long double)R[i]-(long double)S[i]; num+=d*d; long double r=(long double)S[i]; den+=r*r; } return (double)(std::sqrt(num)/(std::sqrt(den)+1e-30L)); };
                double rel_vs_pop_ref = rel(C_pop, C_pop_ref);
                double rel_vs_orig_ref = C_orig_ref.empty() ? -1.0 : rel(C_pop, C_orig_ref);
                if (should_log(2))
                {
                    LOG_INFO("[CosmaPrefill][deep] post rel_l2(C vs pop_ref)=" << rel_vs_pop_ref << " rel_l2(C vs orig_ref)=" << rel_vs_orig_ref);
                }
            }
        }
        // Optional correctness comparison against replicated local GEMM (rank 0) for debugging.
        if (compare_replicated)
        {
            std::vector<float> C_direct((size_t)m * n, 0.f);
            to_row_major(C, C_direct.data());
            if (rank_ == 0)
            {
                const float *A_ptr = A_compat.original_row_major ? A_compat.original_row_major : A.original_row_major;
                const float *B_ptr = B_compat.original_row_major ? B_compat.original_row_major : W.view.original_row_major;
                if (A_ptr && B_ptr)
                {
                    std::vector<float> C_ref((size_t)m * n, 0.f);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, transposeW ? CblasTrans : CblasNoTrans,
                                m, n, k, alpha, A_ptr, k, B_ptr, n, beta, C_ref.data(), n);
                    double num = 0.0, den = 0.0, max_abs = 0.0;
                    for (int i = 0; i < m * n; ++i)
                    {
                        double diff = (double)C_direct[i] - (double)C_ref[i];
                        num += diff * diff;
                        double rv = (double)C_ref[i];
                        den += rv * rv;
                        max_abs = std::max(max_abs, std::abs(diff));
                    }
                    double rel_l2 = std::sqrt(num) / (std::sqrt(den) + 1e-30);
                    if (should_log(2))
                    {
                        LOG_INFO("[CosmaPrefill][compare-replicated] rel_l2=" << rel_l2 << " max_abs=" << max_abs);
                    }
                }
                if (B_compat.mat && B_compat.mat->matrix_pointer())
                {
                    float *localB = B_compat.mat->matrix_pointer();
                    size_t szB = B_compat.mat->matrix_size();
                    std::fill(localB, localB + szB, 0.f);
                    bool legacy_forward = pop_forward_legacy;
                    bool dest_local_mode = !legacy_forward;
                    bool trace_scatter_B2 = diag_deep && should_log(4);
                    if (dest_local_mode)
                    {
                        if (trace_scatter_B2)
                            LOG_TRACE("[CosmaPrefill][pop] destination-local population (B) compare path");
                        for (size_t li = 0; li < szB; ++li)
                        {
                            auto gc = B_compat.mat->global_coordinates((int)li);
                            int gi = gc.first;
                            int gj = gc.second; // B dims k x n
                            if (gi >= 0 && gi < k && gj >= 0 && gj < n)
                            {
                                localB[li] = B_full_buf[(size_t)gi * n + gj];
                                if (diag_deep && gi < std::min(k, 8) && gj < std::min(n, 8) && rank_ == 0 && should_log(4))
                                {
                                    LOG_TRACE("[CosmaPrefill][scatterB-dest] li=" << li << " (gi=" << gi << ",gj=" << gj << ") val=" << localB[li]);
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int gi = 0; gi < k; ++gi)
                        {
                            const float *row_ptr = B_full_buf.data() + (size_t)gi * n;
                            for (int gj = 0; gj < n; ++gj)
                            {
                                auto lc = B_compat.mat->local_coordinates(gi, gj);
                                if (lc.second == rank_)
                                {
                                    int lidx = lc.first;
                                    if (lidx >= 0 && (size_t)lidx < szB)
                                        localB[lidx] = row_ptr[gj];
                                    if (diag_deep && gi < std::min(k, 8) && gj < std::min(n, 8) && rank_ == 0 && should_log(4))
                                    {
                                        LOG_TRACE("[CosmaPrefill][scatterB] (gi=" << gi << ",gj=" << gj << ") val=" << row_ptr[gj] << " lidx=" << lidx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Return distributed result
        if (A_compat.mat)
        {
            C.release_chain.push_back(A_compat.mat);
            std::cerr << "[CosmaPrefill][matmul-release-chain] label=" << A_compat.label
                      << " ptr=" << static_cast<void *>(A_compat.mat->matrix_pointer()) << std::endl;
        }
        if (B_compat.mat)
        {
            C.release_chain.push_back(B_compat.mat);
            std::cerr << "[CosmaPrefill][matmul-release-chain] label=" << B_compat.label
                      << " ptr=" << static_cast<void *>(B_compat.mat->matrix_pointer()) << std::endl;
        }
        return C;
    }

    // Convert COSMA view back to row-major; normalize only for non-output tensors.
    void CosmaPrefillManager::to_row_major(const CosmaView &src, float *dst) const
    {
        if (!dst)
            return;
        bool normalize = src.label != 'C';
        reconstruct_matrix(src, dst, normalize);
    }

    void CosmaPrefillManager::reconstruct_matrix(const CosmaView &src, float *dst, bool normalize) const
    {
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        reconstruct_matrix_impl(src, dst, normalize, env);
    }

    void CosmaPrefillManager::reconstruct_matrix_impl(const CosmaView &src, float *dst, bool normalize, const EnvSnapshot &env) const
    {
        size_t total = static_cast<size_t>(src.global_rows) * src.global_cols;
        bool recon_transpose = env.diag_recon_transpose;
        bool brute_recon = env.diag_recon_brute;
        if (world_size_ == 1)
        {
            if (src.host_owned)
            {
                std::memcpy(dst, src.host_owned->data(), total * sizeof(float));
                return;
            }
            if (!src.mat && src.original_row_major)
            {
                std::memcpy(dst, src.original_row_major, total * sizeof(float));
                return;
            }
            if (src.mat)
            {
                float *local = src.mat->matrix_pointer();
                size_t sz = src.mat->matrix_size();
                size_t copy = std::min(sz, total);
                if (local && copy)
                {
                    std::memcpy(dst, local, copy * sizeof(float));
                    if (copy < total)
                        std::fill(dst + copy, dst + total, 0.f);
                }
                else
                {
                    std::fill(dst, dst + total, 0.f);
                }
            }
            else
            {
                std::fill(dst, dst + total, 0.f);
            }
            return;
        }
        if (src.host_owned)
        {
            std::memcpy(dst, src.host_owned->data(), total * sizeof(float));
            MPI_Bcast(dst, static_cast<int>(total), MPI_FLOAT, 0, MPI_COMM_WORLD);
            return;
        }
        if (!src.mat)
        {
            std::fill(dst, dst + total, 0.f);
            MPI_Bcast(dst, static_cast<int>(total), MPI_FLOAT, 0, MPI_COMM_WORLD);
            return;
        }

        if (brute_recon)
        {
            double t0 = MPI_Wtime();
            std::fill(dst, dst + total, 0.f);
            std::vector<int> counts(total, 0);
            float *local_ptr = src.mat->matrix_pointer();
            size_t lsz = src.mat->matrix_size();
            if (local_ptr && lsz)
            {
                for (int gi = 0; gi < src.global_rows; ++gi)
                {
                    for (int gj = 0; gj < src.global_cols; ++gj)
                    {
                        auto lc = src.mat->local_coordinates(gi, gj);
                        int local_index = lc.first;
                        int owner = lc.second;
                        if (owner == rank_ && local_index >= 0 && (size_t)local_index < lsz)
                        {
                            size_t w = (size_t)gi * src.global_cols + (size_t)gj;
                            dst[w] = local_ptr[local_index];
                            counts[w] = 1;
                        }
                    }
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, dst, (int)total, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, counts.data(), (int)total, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (normalize)
            {
                for (size_t i = 0; i < total; ++i)
                {
                    int c = counts[i];
                    if (c > 1)
                        dst[i] /= (float)c;
                }
            }
            double t1 = MPI_Wtime();
            if (rank_ == 0 && should_log(2))
            {
                long long dup = 0;
                for (int c : counts)
                    if (c > 1)
                        ++dup;
                LOG_INFO("[CosmaPrefill][brute-recon] rows=" << src.global_rows << " cols=" << src.global_cols
                                                             << " time_ms=" << (t1 - t0) * 1000.0
                                                             << " duplicates=" << dup);
            }
            return;
        }

        std::vector<float> local_acc(total, 0.f);
        std::vector<int> ownership_counts(total, 0);
        float *local = src.mat->matrix_pointer();
        size_t sz = src.mat->matrix_size();
        bool map_diag = env.diag_recon_map;
        int map_samples = map_diag ? 32 : 0;
        int emitted = 0;
        if (local && sz)
        {
            for (size_t li = 0; li < sz; ++li)
            {
                auto gc = src.mat->global_coordinates(static_cast<int>(li));
                int gi = gc.first;
                int gj = gc.second;
                if (gi >= 0 && gi < src.global_rows && gj >= 0 && gj < src.global_cols)
                {
                    size_t write_index = recon_transpose ? (size_t)gj * src.global_rows + gi
                                                         : (size_t)gi * src.global_cols + gj;
                    if (write_index < total)
                    {
                        local_acc[write_index] += local[li];
                        ownership_counts[write_index] += 1;
                    }
                }
                if (map_diag && emitted < map_samples && ((li & 127) == 0 || li < (size_t)map_samples))
                {
                    auto lc = src.mat->local_coordinates(gi, gj);
                    int back_li = lc.first;
                    bool present = lc.second;
                    if (rank_ == 0 && should_log(4))
                    {
                        LOG_TRACE("[CosmaPrefill][recon-map] li=" << li
                                                                  << " -> (gi=" << gi << ",gj=" << gj << ") back_li=" << back_li
                                                                  << " present=" << present
                                                                  << " recon_transpose=" << (recon_transpose ? 1 : 0));
                    }
                    ++emitted;
                }
            }
        }

        std::vector<int> global_counts = ownership_counts;
        if (mpi_is_initialized() && mpi_world_size_safe() > 1)
        {
            MPI_Allreduce(local_acc.data(), dst, static_cast<int>(total), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(ownership_counts.data(), global_counts.data(), static_cast<int>(total), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
        else
        {
            std::copy(local_acc.begin(), local_acc.end(), dst);
        }
        ownership_counts.swap(global_counts);
        if (env.diag_deep && rank_ == 0 && should_log(4))
        {
            long long multi_count = 0;
            int max_c = 0;
            int min_c = 1000000;
            long double sum_c = 0.0L;
            long double sum_c2 = 0.0L;
            for (size_t idx = 0; idx < total; ++idx)
            {
                int c = ownership_counts[idx];
                if (c)
                {
                    if (c > 1)
                        ++multi_count;
                    if (c > max_c)
                        max_c = c;
                    if (c < min_c)
                        min_c = c;
                    sum_c += c;
                    sum_c2 += (long double)c * c;
                }
            }
            long double mean_c = sum_c / ((long double)src.global_rows * src.global_cols);
            long double var_c = (sum_c2 / ((long double)src.global_rows * src.global_cols)) - mean_c * mean_c;
            LOG_TRACE("[CosmaPrefill][recon] ownership_counts min=" << min_c << " max=" << max_c << " mean=" << (double)mean_c << " var=" << (double)var_c << " multi=" << multi_count);
        }
        if (env.diag_skip_norm)
        {
            normalize = false;
        }
        if (normalize)
        {
            for (size_t idx = 0; idx < total; ++idx)
            {
                int c = ownership_counts[idx];
                if (c > 1)
                {
                    dst[idx] /= static_cast<float>(c);
                }
            }
        }
    }

    void CosmaPrefillManager::debug_compare_original(const CosmaView &src, int rows, int cols, const float *original) const
    {
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        debug_compare_original_impl(src, rows, cols, original, env);
    }

    void CosmaPrefillManager::debug_compare_original_impl(const CosmaView &src, int rows, int cols, const float *original, const EnvSnapshot &env) const
    {
        if (!env.debug_recon)
            return;
        std::vector<float> recon((size_t)rows * cols, 0.f);
        reconstruct_matrix_impl(src, recon.data(), true, env);
        double num = 0.0, den = 0.0, max_abs = 0.0;
        size_t total = (size_t)rows * cols;
        for (size_t i = 0; i < total; ++i)
        {
            double diff = (double)recon[i] - (double)original[i];
            num += diff * diff;
            den += (double)original[i] * (double)original[i];
            max_abs = std::max(max_abs, std::abs(diff));
        }
        double rel_l2 = std::sqrt(num) / (std::sqrt(den) + 1e-30);
        if (rank_ == 0 && should_log(3))
        {
            LOG_DEBUG("[CosmaPrefill][Recon] rel_l2=" << rel_l2 << " max_abs=" << max_abs << " rows=" << rows << " cols=" << cols);
        }
    }

    void CosmaPrefillManager::diag_global_checksum(const CosmaView &src, const char *tag) const
    {
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        diag_global_checksum_impl(src, tag, env);
    }

    void CosmaPrefillManager::diag_global_checksum_impl(const CosmaView &src, const char *tag, const EnvSnapshot &env) const
    {
        if (!env.diag_enabled)
            return;
        double local_sum = 0.0, local_sq = 0.0;
        size_t local_count = 0;
        if (src.mat && src.mat->matrix_pointer())
        {
            float *ptr = src.mat->matrix_pointer();
            size_t sz = src.mat->matrix_size();
            for (size_t i = 0; i < sz; ++i)
            {
                double v = ptr[i];
                local_sum += v;
                local_sq += v * v;
            }
            local_count = sz;
        }
        else if (src.original_row_major && src.label != 'C')
        {
            size_t sz = (size_t)src.global_rows * src.global_cols;
            for (size_t i = 0; i < sz; ++i)
            {
                double v = src.original_row_major[i];
                local_sum += v;
                local_sq += v * v;
            }
            local_count = sz;
        }
        double global[3] = {local_sum, local_sq, (double)local_count};
        if (mpi_is_initialized() && mpi_world_size_safe() > 1)
        {
            MPI_Allreduce(MPI_IN_PLACE, global, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        if (rank_ == 0 && should_log(3))
        {
            double mean = global[2] ? global[0] / global[2] : 0.0;
            double rms = global[2] ? std::sqrt(global[1] / global[2]) : 0.0;
            LOG_DEBUG("[CosmaPrefill][Diag] checksum tag=" << tag << " mean=" << mean << " rms=" << rms << " count=" << (long long)global[2]);
        }
    }

    void CosmaPrefillManager::diag_sample_points(const CosmaView &src, const float *original, const char *tag) const
    {
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        diag_sample_points_impl(src, original, tag, env);
    }

    void CosmaPrefillManager::diag_sample_points_impl(const CosmaView &src, const float *original, const char *tag, const EnvSnapshot &env) const
    {
        if (!env.diag_enabled || !src.mat || !env.diag_samples_active)
            return;
        const std::string &s = env.diag_samples_spec;
        std::vector<std::pair<int, int>> pts;
        size_t pos = 0;
        while (pos < s.size())
        {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos)
                break;
            size_t semi = s.find(';', comma + 1);
            int r = std::atoi(s.substr(pos, comma - pos).c_str());
            int c = std::atoi(s.substr(comma + 1, (semi == std::string::npos ? s.size() : semi) - (comma + 1)).c_str());
            pts.emplace_back(r, c);
            if (semi == std::string::npos)
                break;
            pos = semi + 1;
        }
        for (auto &p : pts)
        {
            int gi = p.first, gj = p.second;
            if (gi >= src.global_rows || gj >= src.global_cols)
                continue;
            auto lc = src.mat->local_coordinates(gi, gj);
            int owner = lc.second;
            int lidx = lc.first;
            float recon_val = 0.f;
            if (owner == rank_ && lidx >= 0 && src.mat->matrix_pointer() && (size_t)lidx < src.mat->matrix_size())
            {
                recon_val = src.mat->matrix_pointer()[lidx];
            }
            if (mpi_is_initialized() && mpi_world_size_safe() > 1)
            {
                MPI_Bcast(&recon_val, 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
            }
            float orig_val = original ? original[(size_t)gi * src.global_cols + gj] : 0.f;
            if (rank_ == 0 && should_log(4))
            {
                LOG_TRACE("[CosmaPrefill][DiagSample] tag=" << tag << " (" << gi << "," << gj << ") val=" << recon_val << " orig=" << orig_val << " owner=" << owner);
            }
        }
    }

    void CosmaPrefillManager::diag_compare_row_major(const float *A, const float *B, int m, int n, const char *tagA, const char *tagB) const
    {
        EnvSnapshot env_storage;
        const auto &env = resolve_env(nullptr, env_storage);
        diag_compare_row_major_impl(A, B, m, n, tagA, tagB, env);
    }

    void CosmaPrefillManager::diag_compare_row_major_impl(const float *A, const float *B, int m, int n, const char *tagA, const char *tagB, const EnvSnapshot &env) const
    {
        if (!env.diag_enabled || !A || !B)
            return;
        long double num = 0.0L, den = 0.0L, max_abs = 0.0L;
        size_t total = (size_t)m * n;
        for (size_t i = 0; i < total; ++i)
        {
            long double diff = (long double)A[i] - (long double)B[i];
            num += diff * diff;
            long double ref = (long double)A[i];
            den += ref * ref;
            max_abs = std::max(max_abs, fabsl(diff));
        }
        long double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30L);
        if (rank_ == 0 && should_log(2))
        {
            LOG_INFO("[CosmaPrefill][DiagGEMM] compare " << tagA << " vs " << tagB << " rel_l2=" << (double)rel << " max_abs=" << (double)max_abs << " m=" << m << " n=" << n);
        }
    }

    void CosmaPrefillManager::release_weight(CosmaWeightHandle &&handle)
    {
        float *mat_ptr = handle.view.mat ? handle.view.mat->matrix_pointer() : nullptr;
        if (mat_ptr)
        {
            std::cerr << "[CosmaPrefill][release_weight] label=" << handle.desc.id
                      << " ptr=" << static_cast<void *>(mat_ptr) << std::endl;
        }

        long long bytes = 0;
        if (handle.view.mat && mat_ptr)
        {
            bytes = (long long)handle.view.mat->matrix_size() * (long long)sizeof(float);
            handle.view.mat.reset();
        }

        for (auto it = handle.view.release_chain.rbegin(); it != handle.view.release_chain.rend(); ++it)
        {
            it->reset();
        }
        handle.view.release_chain.clear();

        if (bytes > 0)
        {
            stats_.current_resident_bytes -= bytes;
        }
        recalc_resident();
    }
    bool CosmaPrefillManager::memory_budget_allows(size_t bytes_needed) const
    {
        long long limit_bytes = max_resident_mb_ * 1024ll * 1024ll;
        if (limit_bytes <= 0)
            return true;
        long long request = static_cast<long long>(bytes_needed);
        if (request > limit_bytes)
        {
            const_cast<CosmaPrefillManager *>(this)->stats_.allocations_denied++;
            if (rank_ == 0 && should_log(1))
            {
                LOG_WARN("[CosmaPrefill] Denying allocation exceeding budget bytes=" << (request / (1024.0 * 1024.0))
                                                                                     << " MB budget=" << max_resident_mb_ << " MB");
            }
            return false;
        }
        // Recalculate to prune expired weak_ptrs before evaluating cumulative usage.
        const_cast<CosmaPrefillManager *>(this)->recalc_resident();
        long long current = stats_.current_resident_bytes.load();
        if (current + request > limit_bytes)
        {
            const_cast<CosmaPrefillManager *>(this)->stats_.allocations_denied++;
            if (rank_ == 0 && should_log(1))
            {
                LOG_WARN("[CosmaPrefill] Denying allocation due to cumulative budget bytes_current="
                         << (current / (1024.0 * 1024.0)) << " MB request=" << (request / (1024.0 * 1024.0))
                         << " MB budget=" << max_resident_mb_ << " MB");
            }
            return false;
        }
        return true;
    }

    void CosmaPrefillManager::maybe_validation_tile_gemm(const float *A, const float *B, const float *C_full,
                                                         int m, int k, int n)
    {
        if (validate_tile_tokens_ <= 0 || world_size_ <= 1 || !mpi_is_initialized())
            return; // Only meaningful multi-rank
        int tile_m = std::min(validate_tile_tokens_, m);
        int tile_n = std::min(validate_tile_tokens_, n);
        if (tile_m <= 0 || tile_n <= 0)
            return;
        stats_.validation_tile_checks++;
        if (rank_ != 0)
            return; // Only rank 0 performs comparison & logs

        // Compute reference tile C_ref_tile = A_tile * B (first tile_m rows, first tile_n cols)
        std::vector<float> C_ref(tile_m * tile_n, 0.f);
        // We only need first tile_m rows of A and all k columns, and first tile_n columns of B.
        // Row-major GEMM for the tile: naive or cblas
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tile_m, tile_n, k,
                    1.0f,
                    A, k,
                    B, n,
                    0.0f,
                    C_ref.data(), tile_n);

        // Extract corresponding region from C_full (row-major m x n)
        double num = 0.0, den = 0.0, max_abs = 0.0;
        for (int i = 0; i < tile_m; ++i)
        {
            const float *c_row = C_full + (size_t)i * n;
            for (int j = 0; j < tile_n; ++j)
            {
                double diff = (double)c_row[j] - (double)C_ref[i * tile_n + j];
                num += diff * diff;
                double refv = (double)C_ref[i * tile_n + j];
                den += refv * refv;
                max_abs = std::max(max_abs, std::abs(diff));
            }
        }
        double rel_l2 = std::sqrt(num) / std::sqrt(den + 1e-30);
        if (should_log(3))
        {
            LOG_DEBUG("[CosmaPrefill] validation tile m=" << tile_m << " n=" << tile_n
                                                          << " rel_l2=" << rel_l2 << " max_abs=" << max_abs);
        }
        // Simple anomaly heuristic: log warning if rel_l2 exceeds 1e-3 (tight) or NaN
        if (!(rel_l2 == rel_l2) || rel_l2 > 1e-3)
        {
            LOG_WARN("[CosmaPrefill] validation tile anomaly rel_l2=" << rel_l2 << " max_abs=" << max_abs);
        }
    }

} // namespace llaminar

// ===================== Phase 1b Additions (Implementation) =====================
namespace llaminar
{

    void CosmaPrefillManager::dump_stats_json(const std::string &path) const
    {
        if (rank_ != 0)
            return; // only rank 0 writes aggregate file
        std::ofstream ofs(path);
        if (!ofs.is_open())
        {
            if (should_log(1))
                LOG_WARN("[CosmaPrefill][stats] Failed to open output path: " << path);
            return;
        }
        auto q = [&](const char *k, long long v, bool trailing = true)
        { ofs << "  \"" << k << "\": " << v << (trailing ? ",\n" : "\n"); };
        ofs << "{\n";
        q("single_rank_calls", stats_.single_rank_calls.load());
        q("fast_path_calls", stats_.fast_path_calls.load());
        q("cosma_path_calls", stats_.cosma_path_calls.load());
        q("bytes_streamed_weights", stats_.bytes_streamed_weights.load());
        q("bytes_converted_activations", stats_.bytes_converted_activations.load());
        q("matmul_invocations", stats_.matmul_invocations.load());
        q("validation_tile_checks", stats_.validation_tile_checks.load());
        q("us_stream_weights", stats_.us_stream_weights.load());
        q("us_convert_activation", stats_.us_convert_activation.load());
        q("us_matmul", stats_.us_matmul.load());
        q("strategy_cache_hits", strategy_cache_.stats.hits.load());
        q("strategy_cache_misses", strategy_cache_.stats.misses.load());
        q("current_resident_bytes", stats_.current_resident_bytes.load());
        q("peak_resident_bytes", stats_.peak_resident_bytes.load());
        q("allocations_tracked", stats_.allocations_tracked.load());
        q("allocations_denied", stats_.allocations_denied.load());
        q("preflight_invocations", stats_.preflight_invocations.load());
        q("preflight_denied", stats_.preflight_denied.load());
        q("preflight_estimated_bytes_last", stats_.preflight_estimated_bytes_last.load());
        q("fused_dequant_invocations", stats_.fused_dequant_invocations.load());
        q("fused_dequant_elements", stats_.fused_dequant_elements.load());
        q("softmax_invocations", stats_.softmax_invocations.load());
        q("softmax_rows_processed", stats_.softmax_rows_processed.load());
        q("us_softmax", stats_.us_softmax.load());
        q("fused_rmsnorm_qkv_invocations", stats_.fused_rmsnorm_qkv_invocations.load());
        q("us_fused_rmsnorm_qkv", stats_.us_fused_rmsnorm_qkv.load());
        q("overlap_stream_invocations", stats_.overlap_stream_invocations.load());
        q("us_overlap_stream", stats_.us_overlap_stream.load(), false);
        ofs << "}\n";
        if (should_log(3))
            LOG_DEBUG("[CosmaPrefill][stats] Wrote JSON stats to " << path);
    }

    void CosmaPrefillManager::dump_stats_if_requested() const
    {
        const char *flag = std::getenv("LLAMINAR_COSMA_DUMP_STATS");
        if (!flag)
            return;
        auto should_emit = [](const char *value) -> bool
        {
            if (!value)
                return false;
            std::string token;
            token.reserve(std::strlen(value));
            for (const char *p = value; *p; ++p)
            {
                if (!std::isspace(static_cast<unsigned char>(*p)))
                    token.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
            }
            if (token.empty())
                return true; // treat empty as enabled when var present
            return !(token == "0" || token == "false" || token == "off" || token == "no");
        };
        if (!should_emit(flag))
            return;
        std::string path = "cosma_prefill_stats.json";
        if (const char *env_path = std::getenv("LLAMINAR_COSMA_DUMP_STATS_PATH"))
        {
            std::string candidate = env_path;
            if (!candidate.empty())
                path = std::move(candidate);
        }
        dump_stats_json(path);
    }

    void CosmaPrefillManager::reset_stats()
    {
        stats_.single_rank_calls = 0;
        stats_.fast_path_calls = 0;
        stats_.cosma_path_calls = 0;
        stats_.bytes_streamed_weights = 0;
        stats_.bytes_converted_activations = 0;
        stats_.matmul_invocations = 0;
        stats_.validation_tile_checks = 0;
        stats_.us_stream_weights = 0;
        stats_.us_convert_activation = 0;
        stats_.us_matmul = 0;
        stats_.current_resident_bytes = 0;
        stats_.peak_resident_bytes = 0;
        stats_.allocations_tracked = 0;
        stats_.allocations_denied = 0;
        stats_.preflight_invocations = 0;
        stats_.preflight_denied = 0;
        stats_.preflight_estimated_bytes_last = 0;
        stats_.fused_dequant_invocations = 0;
        stats_.fused_dequant_elements = 0;
        stats_.softmax_invocations = 0;
        stats_.softmax_rows_processed = 0;
        stats_.us_softmax = 0;
        stats_.fused_rmsnorm_qkv_invocations = 0;
        stats_.us_fused_rmsnorm_qkv = 0;
        stats_.overlap_stream_invocations = 0;
        stats_.us_overlap_stream = 0;
        strategy_cache_.stats.hits = 0;
        strategy_cache_.stats.misses = 0;
        {
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            allocations_.clear();
        }
    }

    const std::vector<std::string> &CosmaPrefillManager::recognized_env_vars()
    {
        static std::vector<std::string> vars = {
            "ADAPTIVE_DISABLE_COSMA",
            "LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE",
            "LLAMINAR_COSMA_COMPARE_REPLICATED",
            "LLAMINAR_COSMA_DEBUG_RECON",
            "LLAMINAR_COSMA_DIAG",
            "LLAMINAR_COSMA_DIAG_AXIS",
            "LLAMINAR_COSMA_DIAG_COORD_INVERT",
            "LLAMINAR_COSMA_DIAG_DEEP",
            "LLAMINAR_COSMA_DIAG_LOCAL_PROBE",
            "LLAMINAR_COSMA_DIAG_LOCAL_PROBE_DEEP",
            "LLAMINAR_COSMA_DIAG_PERM_INFER",
            "LLAMINAR_COSMA_DIAG_RECON_BYPASS",
            "LLAMINAR_COSMA_DIAG_RECON_BRUTE",
            "LLAMINAR_COSMA_DIAG_RECON_MAP",
            "LLAMINAR_COSMA_DIAG_RECON_TRANSPOSE",
            "LLAMINAR_COSMA_DIAG_SKIP_NORM",
            "LLAMINAR_COSMA_DIAG_SAMPLES",
            "LLAMINAR_COSMA_DIAG_SWAPRC",
            "LLAMINAR_COSMA_DIAG_TAP",
            "LLAMINAR_COSMA_DIAG_TRY_TRANSPOSE",
            "LLAMINAR_COSMA_DISABLE",
            "LLAMINAR_COSMA_DISABLE_FUSED_DEQUANT",
            "LLAMINAR_COSMA_DIRECT_THRESHOLD_OPS",
            "LLAMINAR_COSMA_DUMP_SMALL",
            "LLAMINAR_COSMA_DUMP_STATS",
            "LLAMINAR_COSMA_DUMP_STATS_PATH",
            "LLAMINAR_COSMA_FAST_PATH_THRESHOLD",
            "LLAMINAR_COSMA_FAST_UNVERIFIED",
            "LLAMINAR_COSMA_FORCE",
            "LLAMINAR_COSMA_FORCE_DIRECT",
            "LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT",
            "LLAMINAR_COSMA_FORCE_REPLICATED",
            "LLAMINAR_COSMA_FORCE_REPLICATED_DIAG",
            "LLAMINAR_COSMA_FORCE_REPLICATED_THREADS",
            "LLAMINAR_COSMA_FORCE_UNIFIED",
            "LLAMINAR_COSMA_LOG_LEVEL",
            "LLAMINAR_COSMA_MAX_RESIDENT_MB",
            "LLAMINAR_COSMA_OVERLAP_STREAM",
            "LLAMINAR_COSMA_OVERLAP_VERBOSE",
            "LLAMINAR_COSMA_POP_FORWARD_LEGACY",
            "LLAMINAR_COSMA_PREFLIGHT_DISABLE",
            "LLAMINAR_COSMA_PREFLIGHT_SAFETY_FACTOR",
            "LLAMINAR_COSMA_PREFILL_THRESHOLD",
            "LLAMINAR_COSMA_RECON_FORCE_LEGACY",
            "LLAMINAR_COSMA_REPLICATE_B",
            "LLAMINAR_COSMA_TEST_TRACE",
            "LLAMINAR_COSMA_VALIDATE_TILE",
            "LLAMINAR_OPENBLAS_THREADS",
            "LLAMINAR_SKIP_MPI_IN_SINGLE_TEST",
            "OPENBLAS_NUM_THREADS"};
        return vars;
    }

    void CosmaPrefillManager::recalc_resident()
    {
        long long sum = 0;
        {
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            for (auto &rec : allocations_)
            {
                if (rec.ref.expired())
                    continue;
                sum += rec.bytes;
            }
        }
        stats_.current_resident_bytes = sum;
        long long peak = stats_.peak_resident_bytes.load();
        if (sum > peak)
            stats_.peak_resident_bytes = sum;
    }

    bool CosmaPrefillManager::preflight_allows(int seq_len, int d_model, int d_ff, int n_proj) const
    {
        // Skip if disabled by env or single-rank trivial case
        if (std::getenv("LLAMINAR_COSMA_PREFLIGHT_DISABLE") || world_size_ == 1)
            return true;
        const double safety = [&]()
        {
            if (const char *s = std::getenv("LLAMINAR_COSMA_PREFLIGHT_SAFETY_FACTOR"))
            {
                try
                {
                    return std::max(0.5, std::min(4.0, std::stod(s)));
                }
                catch (...)
                {
                    return 1.2;
                }
            }
            return 1.2; // default 20% headroom
        }();
        // Estimate major working set components for a typical attention+MLP block:
        //  - Activation input (seq_len * d_model)
        //  - Q,K,V projections intermediates (n_proj * seq_len * d_model) (reuse of activation layout not counted separately)
        //  - Attention scores (seq_len * seq_len) transient (worst-case; large!)
        //  - MLP intermediates (seq_len * d_ff * 2 for gate+up)
        //  - Output buffers (seq_len * d_model) * 2 (residual staging)
        long long bytes_activation = 1ll * seq_len * d_model * sizeof(float);
        long long bytes_qkv = 1ll * n_proj * seq_len * d_model * sizeof(float);
        long long bytes_scores = 1ll * seq_len * seq_len * sizeof(float); // may be tiled later
        long long bytes_mlp = 2ll * seq_len * d_ff * sizeof(float);
        long long bytes_output = 2ll * seq_len * d_model * sizeof(float);
        long long estimate = bytes_activation + bytes_qkv + bytes_scores + bytes_mlp + bytes_output;
        estimate = (long long)(estimate * safety);
        // Update stats (mutable in const via cast because instrumentation)
        const_cast<CosmaPrefillManager *>(this)->stats_.preflight_invocations++;
        const_cast<CosmaPrefillManager *>(this)->stats_.preflight_estimated_bytes_last = estimate;
        long long budget_bytes = max_resident_mb_ * 1024ll * 1024ll;
        if (estimate > budget_bytes)
        {
            if (rank_ == 0 && should_log(1))
            {
                LOG_WARN("[CosmaPrefill][preflight] Estimated working set " << (estimate / (1024.0 * 1024.0))
                                                                            << " MB exceeds budget "
                                                                            << max_resident_mb_ << " MB (seq=" << seq_len
                                                                            << ", d_model=" << d_model << ", d_ff=" << d_ff << ")");
            }
            const_cast<CosmaPrefillManager *>(this)->stats_.preflight_denied++;
            return false;
        }
        if (rank_ == 0 && should_log(3))
        {
            LOG_DEBUG("[CosmaPrefill][preflight] estimate=" << (estimate / (1024.0 * 1024.0))
                                                            << " MB within budget " << max_resident_mb_ << " MB");
        }
        return true;
    }

    // ================= In-layout Elementwise (Phase 2) =================
    bool CosmaPrefillManager::rmsnorm_in_layout(const CosmaView &src, CosmaView &dst, const float *weight, int seq_len, int hidden_size, float eps)
    {
        if (!weight || hidden_size <= 0 || seq_len <= 0)
            return false;
        // Single-rank row-major fallback (no COSMA allocation)
        if (world_size_ == 1 && (!src.mat || !dst.mat) && src.original_row_major && dst.original_row_major)
        {
            const float *in = src.original_row_major;
            float *out = const_cast<float *>(dst.original_row_major);
            for (int r = 0; r < seq_len; ++r)
            {
                const float *row = in + (size_t)r * hidden_size;
                double sum = 0.0;
                for (int c = 0; c < hidden_size; ++c)
                {
                    double v = row[c];
                    sum += v * v;
                }
                double inv = 1.0 / std::sqrt(sum / hidden_size + eps);
                float *dst_row = out + (size_t)r * hidden_size;
                for (int c = 0; c < hidden_size; ++c)
                {
                    dst_row[c] = float(row[c] * inv * weight[c]);
                }
            }
            return true;
        }
        if (!src.mat || !dst.mat)
        {
            if (should_log(4))
            {
                LOG_DEBUG("[CosmaPrefill][rmsnorm] distributed path abort: missing mat src=" << (src.mat ? 1 : 0) << " dst=" << (dst.mat ? 1 : 0) << " rank=" << rank_);
            }
            return false; // multi-rank requires distributed matrices
        }
        float *dst_local = dst.mat->matrix_pointer();
        const float *src_local = src.mat->matrix_pointer();
        size_t local_size = src.mat->matrix_size();
        if (!dst_local || !src_local)
        {
            // Some ranks may legitimately own no tiles (matrix_size()==0) -> treat as no-op success.
            if (local_size == 0)
            {
                if (should_log(5))
                {
                    LOG_TRACE("[CosmaPrefill][rmsnorm] rank " << rank_ << " owns no tiles (size=0) -> no-op success");
                }
                return true;
            }
            if (should_log(4))
            {
                LOG_DEBUG("[CosmaPrefill][rmsnorm] distributed path abort: null local pointer despite local_size>0 src_local=" << (void *)src_local << " dst_local=" << (void *)dst_local << " local_size=" << local_size << " rank=" << rank_);
            }
            return false;
        }
        if (local_size == 0)
            return true; // nothing owned on this rank
        auto mat_ptr = src.mat.get();
        // Gather local indices grouped by row (sequence position).
        std::vector<std::vector<int>> row_indices(seq_len);
        row_indices.reserve(seq_len);
        for (size_t li = 0; li < local_size; ++li)
        {
            auto g = mat_ptr->global_coordinates(static_cast<int>(li));
            int gi = g.first;
            int gj = g.second;
            if ((unsigned)gi < (unsigned)seq_len && (unsigned)gj < (unsigned)hidden_size)
                row_indices[gi].push_back(static_cast<int>(li));
        }
#pragma omp parallel for schedule(static)
        for (int r = 0; r < seq_len; ++r)
        {
            auto &idxs = row_indices[r];
            if (idxs.empty())
                continue;
            double sum_sq = 0.0;
            for (int li : idxs)
            {
                float v = src_local[li];
                sum_sq += double(v) * double(v);
            }
            double inv_scale = 1.0 / std::sqrt(sum_sq / std::max(1, (int)idxs.size()) + eps);
            for (int li : idxs)
            {
                int gj = mat_ptr->global_coordinates(li).second;
                float gamma = (gj >= 0 && gj < hidden_size) ? weight[gj] : 1.f;
                dst_local[li] = float(src_local[li] * inv_scale * gamma);
            }
        }
        return true;
    }

    bool CosmaPrefillManager::swiglu_in_layout(const CosmaView &gate, const CosmaView &up, CosmaView &out, int seq_len, int hidden_size)
    {
        if (hidden_size <= 0 || seq_len <= 0)
            return false;
        if (world_size_ == 1 && (!gate.mat || !up.mat || !out.mat) && gate.original_row_major && up.original_row_major && out.original_row_major)
        {
            const float *gptr = gate.original_row_major;
            const float *uptr = up.original_row_major;
            float *optr = const_cast<float *>(out.original_row_major);
            auto silu = [](float x) -> float
            { return x / (1.0f + std::exp(-x)); };
            size_t stride = (size_t)hidden_size;
            for (int r = 0; r < seq_len; ++r)
            {
                const float *grow = gptr + r * stride;
                const float *urow = uptr + r * stride;
                float *orow = optr + r * stride;
                for (int c = 0; c < hidden_size; ++c)
                {
                    orow[c] = silu(urow[c]) * grow[c];
                }
            }
            return true;
        }
        if (!gate.mat || !up.mat || !out.mat)
            return false;
        const float *g_local = gate.mat->matrix_pointer();
        const float *u_local = up.mat->matrix_pointer();
        float *o_local = out.mat->matrix_pointer();
        size_t local_size = gate.mat->matrix_size();
        if (!g_local || !u_local || !o_local)
        {
            if (local_size == 0)
            {
                if (should_log(5))
                {
                    LOG_TRACE("[CosmaPrefill][swiglu] rank " << rank_ << " owns no tiles (size=0) -> no-op success");
                }
                return true;
            }
            if (should_log(4))
            {
                LOG_DEBUG("[CosmaPrefill][swiglu] abort: null local pointer with local_size=" << local_size << " rank=" << rank_);
            }
            return false;
        }
        if (up.mat->matrix_size() != local_size || out.mat->matrix_size() != local_size)
            return false; // shape mismatch on local partitions
        auto mat_ptr = gate.mat.get();
        std::vector<std::vector<int>> row_indices(seq_len);
        for (size_t li = 0; li < local_size; ++li)
        {
            auto g = mat_ptr->global_coordinates(static_cast<int>(li));
            int gi = g.first;
            int gj = g.second;
            if ((unsigned)gi < (unsigned)seq_len && (unsigned)gj < (unsigned)hidden_size)
                row_indices[gi].push_back(static_cast<int>(li));
        }
        auto silu = [](float x) -> float
        { return x / (1.0f + std::exp(-x)); };
#pragma omp parallel for schedule(static)
        for (int r = 0; r < seq_len; ++r)
        {
            auto &idxs = row_indices[r];
            for (int li : idxs)
            {
                float u = u_local[li];
                float gval = g_local[li];
                o_local[li] = silu(u) * gval;
            }
        }
        return true;
    }

    bool CosmaPrefillManager::softmax_in_layout(const CosmaView &scores, CosmaView &dst, int rows, int cols, float scale)
    {
        if (rows <= 0 || cols <= 0)
            return false;
        auto t0 = std::chrono::high_resolution_clock::now();
        if (world_size_ == 1 && (!scores.mat || !dst.mat) && scores.original_row_major && dst.original_row_major)
        {
            const float *src = scores.original_row_major;
            float *out = const_cast<float *>(dst.original_row_major);
            for (int r = 0; r < rows; ++r)
            {
                const float *row_ptr = src + (size_t)r * cols;
                float *out_row = out + (size_t)r * cols;
                float row_max = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < cols; ++c)
                {
                    row_max = std::max(row_max, row_ptr[c] * scale);
                }
                float denom = 0.f;
                for (int c = 0; c < cols; ++c)
                {
                    float val = std::exp(row_ptr[c] * scale - row_max);
                    out_row[c] = val;
                    denom += val;
                }
                float inv = denom > 0.f ? 1.f / denom : 1.f;
                for (int c = 0; c < cols; ++c)
                {
                    out_row[c] *= inv;
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            stats_.softmax_invocations++;
            stats_.softmax_rows_processed += rows;
            stats_.us_softmax += (long long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            return true;
        }
        if (!scores.mat || !dst.mat)
            return false;
        float *dst_local = dst.mat->matrix_pointer();
        const float *src_local = scores.mat->matrix_pointer();
        size_t local_size = scores.mat->matrix_size();
        if (dst.mat->matrix_size() != local_size)
            return false;
        if ((!src_local || !dst_local) && local_size > 0)
            return false;
        auto mat_ptr = scores.mat.get();
        std::vector<std::vector<int>> row_indices(rows);
        if (local_size > 0)
        {
            for (size_t li = 0; li < local_size; ++li)
            {
                auto g = mat_ptr->global_coordinates(static_cast<int>(li));
                int gi = g.first;
                int gj = g.second;
                if ((unsigned)gi < (unsigned)rows && (unsigned)gj < (unsigned)cols)
                    row_indices[gi].push_back((int)li);
            }
        }
        if (should_log(4))
        {
            LOG_TRACE("[CosmaPrefill][softmax] rank=" << rank_ << " built row_indices local_size=" << local_size);
        }
        if (should_log(3))
        {
            LOG_DEBUG("[CosmaPrefill][softmax] rank=" << rank_ << " local_size=" << local_size << " rows=" << rows);
        }
        std::vector<float> row_max(rows, -std::numeric_limits<float>::infinity());
        for (int r = 0; r < rows; ++r)
        {
            auto &idxs = row_indices[r];
            for (int li : idxs)
            {
                float val = src_local[li] * scale;
                if (val > row_max[r])
                    row_max[r] = val;
            }
        }
        if (should_log(4))
        {
            LOG_TRACE("[CosmaPrefill][softmax] rank=" << rank_ << " row_max local stage done");
        }
        safe_allreduce_float_inplace(row_max.data(), rows, MPI_MAX);
        if (should_log(4))
        {
            LOG_TRACE("[CosmaPrefill][softmax] rank=" << rank_ << " row_max reduced");
        }
        std::vector<float> row_sum(rows, 0.f);
        for (int r = 0; r < rows; ++r)
        {
            auto &idxs = row_indices[r];
            float max_val = row_max[r];
            for (int li : idxs)
            {
                float val = std::exp(src_local[li] * scale - max_val);
                dst_local[li] = val;
                row_sum[r] += val;
            }
        }
        if (should_log(4))
        {
            LOG_TRACE("[CosmaPrefill][softmax] rank=" << rank_ << " row_sum local stage done");
        }
        safe_allreduce_float_inplace(row_sum.data(), rows, MPI_SUM);
        if (should_log(4))
        {
            LOG_TRACE("[CosmaPrefill][softmax] rank=" << rank_ << " row_sum reduced");
        }
        for (int r = 0; r < rows; ++r)
        {
            auto &idxs = row_indices[r];
            float denom = row_sum[r] > 0.f ? row_sum[r] : 1.f;
            for (int li : idxs)
            {
                dst_local[li] /= denom;
            }
        }
        if (should_log(4))
        {
            LOG_TRACE("[CosmaPrefill][softmax] rank=" << rank_ << " normalization done");
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        stats_.softmax_invocations++;
        stats_.softmax_rows_processed += rows;
        stats_.us_softmax += (long long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        return true;
    }

    FusedRmsnormQkvResult CosmaPrefillManager::fused_rmsnorm_qkv(const float *activation_row_major,
                                                                 const float *gamma,
                                                                 const WeightDescriptor &wq,
                                                                 const WeightDescriptor &wk,
                                                                 const WeightDescriptor &wv,
                                                                 int seq_len,
                                                                 int hidden_size,
                                                                 float eps,
                                                                 float softmax_scale,
                                                                 bool transpose_k)
    {
        (void)softmax_scale; // reserved for future integration with softmax scheduling
        FusedRmsnormQkvResult result;
        if (!activation_row_major || !gamma || seq_len <= 0 || hidden_size <= 0)
            return result;
        auto t0 = std::chrono::high_resolution_clock::now();
        const auto &act_strat = strategy_cache_.get(seq_len, hidden_size, hidden_size, world_size_);
        auto act_view = convert_activation_in_with_strategy(activation_row_major, seq_len, hidden_size, act_strat);
        if (act_view.mat)
        {
            std::cerr << "[FusedRmsnormQkv][act_view] ptr=" << static_cast<void *>(act_view.mat->matrix_pointer()) << std::endl;
        }
        CosmaView norm_view;
        if (world_size_ == 1)
        {
            norm_view.global_rows = seq_len;
            norm_view.global_cols = hidden_size;
            norm_view.label = 'A';
            auto buffer = std::make_shared<std::vector<float>>((size_t)seq_len * hidden_size, 0.f);
            norm_view.host_owned = buffer;
            norm_view.original_row_major = buffer->data();
        }
        else
        {
            norm_view = allocate_matrix('A', seq_len, hidden_size, act_strat, false);
            if (norm_view.mat)
            {
                std::cerr << "[FusedRmsnormQkv][norm_view] ptr=" << static_cast<void *>(norm_view.mat->matrix_pointer()) << std::endl;
            }
        }
        if (!rmsnorm_in_layout(act_view, norm_view, gamma, seq_len, hidden_size, eps))
        {
            return result;
        }

        const auto &wq_strat = strategy_cache_.get(wq.rows, wq.cols, wq.rows, world_size_);
        const auto &wk_strat = strategy_cache_.get(wk.rows, wk.cols, wk.rows, world_size_);
        const auto &wv_strat = strategy_cache_.get(wv.rows, wv.cols, wv.rows, world_size_);

        bool overlap_enabled = std::getenv("LLAMINAR_COSMA_OVERLAP_STREAM") != nullptr;
        bool overlap_verbose = std::getenv("LLAMINAR_COSMA_OVERLAP_VERBOSE") != nullptr;
        CosmaWeightHandle wq_handle = load_weight_with_strategy(wq, wq_strat);
        CosmaWeightHandle wk_handle;
        CosmaWeightHandle wv_handle;
        std::future<CosmaWeightHandle> future_wk;
        std::future<CosmaWeightHandle> future_wv;
        std::chrono::high_resolution_clock::time_point overlap_start;
        if (overlap_enabled)
        {
            stats_.overlap_stream_invocations++;
            overlap_start = std::chrono::high_resolution_clock::now();
            auto async_loader = [&](const WeightDescriptor &desc, const cosma::Strategy &strat)
            {
                const cosma::Strategy *sp = &strat;
                return std::async(std::launch::async, [this, desc, sp]()
                                  { return this->load_weight_with_strategy(desc, *sp); });
            };
            future_wk = async_loader(wk, wk_strat);
            future_wv = async_loader(wv, wv_strat);
            if (overlap_verbose && rank_ == 0 && should_log(3))
            {
                LOG_DEBUG("[CosmaPrefill][overlap] launched async weight streaming for K/V");
            }
        }
        else
        {
            wk_handle = load_weight_with_strategy(wk, wk_strat);
            wv_handle = load_weight_with_strategy(wv, wv_strat);
        }

        auto q_view = matmul(norm_view, wq_handle, seq_len, hidden_size, wq.cols);
        if (q_view.mat)
        {
            std::cerr << "[FusedRmsnormQkv][q_view] ptr=" << static_cast<void *>(q_view.mat->matrix_pointer()) << std::endl;
        }

        if (overlap_enabled)
        {
            wk_handle = future_wk.get();
            wv_handle = future_wv.get();
            auto overlap_end = std::chrono::high_resolution_clock::now();
            stats_.us_overlap_stream += (long long)std::chrono::duration_cast<std::chrono::microseconds>(overlap_end - overlap_start).count();
            if (overlap_verbose && rank_ == 0 && should_log(3))
            {
                LOG_DEBUG("[CosmaPrefill][overlap] completed async weight streaming for K/V");
            }
        }

        auto k_view = matmul(norm_view, wk_handle, seq_len, hidden_size, wk.cols, transpose_k);
        if (k_view.mat)
        {
            std::cerr << "[FusedRmsnormQkv][k_view] ptr=" << static_cast<void *>(k_view.mat->matrix_pointer()) << std::endl;
        }
        auto v_view = matmul(norm_view, wv_handle, seq_len, hidden_size, wv.cols);
        if (v_view.mat)
        {
            std::cerr << "[FusedRmsnormQkv][v_view] ptr=" << static_cast<void *>(v_view.mat->matrix_pointer()) << std::endl;
        }

        auto capture_host_owned = [&](CosmaView &view) -> bool
        {
            const size_t elements = static_cast<size_t>(view.global_rows) * view.global_cols;
            if (!view.mat)
            {
                if (!view.host_owned && view.original_row_major && elements > 0)
                {
                    auto buf = std::make_shared<std::vector<float>>(elements, 0.f);
                    std::memcpy(buf->data(), view.original_row_major, elements * sizeof(float));
                    view.host_owned = buf;
                    view.original_row_major = buf->data();
                }
                return true;
            }
            if (world_size_ == 1)
            {
                auto buf = std::make_shared<std::vector<float>>(elements, 0.f);
                reconstruct_matrix(view, buf->data(), view.label != 'C');
                view.host_owned = buf;
                view.original_row_major = buf->data();
                if (view.mat)
                {
                    view.mat.reset();
                }
                for (auto it = view.release_chain.rbegin(); it != view.release_chain.rend(); ++it)
                {
                    it->reset();
                }
                view.release_chain.clear();
                return true;
            }
            // Multi-rank path retains distributed layout; caller will reconstruct on demand.
            return true;
        };

        // Convert distributed outputs to host-owned buffers before returning guards.
        capture_host_owned(v_view);
        capture_host_owned(k_view);
        capture_host_owned(q_view);

        result.wv_guard = FusedRmsnormQkvResult::WeightGuard(this, std::move(wv_handle));
        result.wk_guard = FusedRmsnormQkvResult::WeightGuard(this, std::move(wk_handle));
        result.wq_guard = FusedRmsnormQkvResult::WeightGuard(this, std::move(wq_handle));

        // Normalize buffer must persist for caller; convert to host-owned as well.
        capture_host_owned(norm_view);

        result.activation_guard = act_view;
        result.normalized = norm_view;
        result.q = q_view;
        result.k = k_view;
        result.v = v_view;

        auto t1 = std::chrono::high_resolution_clock::now();
        stats_.fused_rmsnorm_qkv_invocations++;
        stats_.us_fused_rmsnorm_qkv += (long long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        return result;
    }

} // namespace llaminar
