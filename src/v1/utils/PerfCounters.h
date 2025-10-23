#pragma once
#include <atomic>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>
#include <limits>
#include "DebugEnv.h"

namespace llaminar
{

    struct MatmulOpSample
    {
        uint64_t m = 0, n = 0, k = 0; // dimensions
        double ms = 0.0;              // wall time milliseconds
        double gflops = 0.0;          // computed GFLOPS for this op
        int backend = 0;              // backend enum integral (if available)
        uint8_t layer_type = 0;       // 0=unknown,1=MLP,2=ATTN
        uint8_t phase = 0;            // phase within layer (semantic depends on layer_type)
        int32_t layer_index = -1;     // model layer index if known (-1 unknown)
    };

    struct BackendAggStats
    {
        int backend = 0;
        uint64_t count = 0;   // total matmul count
        uint64_t flops = 0;   // raw FLOPs (mul+add) accumulated
        double time_ms = 0.0; // wall time accumulated
        double min_ms = 0.0;  // min latency across all phases
        double max_ms = 0.0;  // max latency across all phases
        // Phase specific aggregates
        uint64_t prefill_count = 0;
        uint64_t decode_count = 0;
        uint64_t prefill_flops = 0;
        uint64_t decode_flops = 0;
        double prefill_time_ms = 0.0;
        double decode_time_ms = 0.0;
        double prefill_min_ms = 0.0;
        double prefill_max_ms = 0.0;
        double decode_min_ms = 0.0;
        double decode_max_ms = 0.0;
    };

    struct CommOpStats
    {
        uint64_t calls = 0;
        uint64_t bytes = 0;
        void record(uint64_t b)
        {
            ++calls;
            bytes += b;
        }
    };

    enum class CommOp : uint8_t
    {
        Bcast,
        Allreduce,
        Allgather,
        Allgatherv,
        Reduce,
        ReduceScatter,
        Alltoall,
        Alltoallv,
        Barrier
    };

    class PerformanceCounters
    {
    public:
        static PerformanceCounters &instance()
        {
            static PerformanceCounters inst;
            return inst;
        }

        // Thread-local phase context for correlating matmuls with higher-level layer phases.
        struct PhaseContext
        {
            uint8_t layer_type = 0;
            uint8_t phase = 0;
            int32_t layer_index = -1;
        };
        static thread_local PhaseContext tl_phase_;
        static void setPhase(uint8_t layer_type, uint8_t phase, int32_t layer_index = -1)
        {
            tl_phase_.layer_type = layer_type;
            tl_phase_.phase = phase;
            // Preserve existing layer index if caller passes -1 (inheritance semantics)
            if (layer_index >= 0)
                tl_phase_.layer_index = layer_index;
        }
        struct PhaseScope
        {
            PhaseContext prev_;
            PhaseScope(uint8_t lt, uint8_t ph, int32_t idx = -1)
            {
                prev_ = tl_phase_;
                setPhase(lt, ph, idx);
            }
            ~PhaseScope() { tl_phase_ = prev_; }
        };

        void record_matmul(uint64_t m, uint64_t n, uint64_t k, double ms, int backend, bool is_prefill)
        {
            const auto &env = debugEnv();
            if (!env.performance.enable)
                return;
            double flops = 2.0 * (double)m * (double)n * (double)k; // mul+add
            double gflops = (ms > 0.0) ? flops / (ms * 1e6) : 0.0;
            total_matmuls_.fetch_add(1, std::memory_order_relaxed);
            total_flops_.fetch_add((uint64_t)flops, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(mu_time_);
                total_time_ms_ += ms;
                if (backend >= 0 && backend < kMaxBackends)
                {
                    backend_counts_[backend] += 1;
                    backend_time_ms_[backend] += ms;
                    backend_flops_[backend] += (uint64_t)flops;
                    backend_min_ms_[backend] = std::min(backend_min_ms_[backend], ms);
                    backend_max_ms_[backend] = std::max(backend_max_ms_[backend], ms);
                    if (is_prefill)
                    {
                        backend_prefill_counts_[backend] += 1;
                        backend_prefill_time_ms_[backend] += ms;
                        backend_prefill_flops_[backend] += (uint64_t)flops;
                        backend_prefill_min_ms_[backend] = std::min(backend_prefill_min_ms_[backend], ms);
                        backend_prefill_max_ms_[backend] = std::max(backend_prefill_max_ms_[backend], ms);
                    }
                    else
                    {
                        backend_decode_counts_[backend] += 1;
                        backend_decode_time_ms_[backend] += ms;
                        backend_decode_flops_[backend] += (uint64_t)flops;
                        backend_decode_min_ms_[backend] = std::min(backend_decode_min_ms_[backend], ms);
                        backend_decode_max_ms_[backend] = std::max(backend_decode_max_ms_[backend], ms);
                    }
                }
            }
            if (env.performance.log_each_matmul && rank_cached_ == env.performance.log_rank)
            {
                std::lock_guard<std::mutex> lk(mu_);
                MatmulOpSample s;
                s.m = m;
                s.n = n;
                s.k = k;
                s.ms = ms;
                s.gflops = gflops;
                s.backend = backend;
                s.layer_type = tl_phase_.layer_type;
                s.phase = tl_phase_.phase;
                s.layer_index = tl_phase_.layer_index;
                samples_.push_back(s);
            }
        }

        void record_comm(CommOp op, uint64_t bytes)
        {
            const auto &env = debugEnv();
            if (!env.performance.enable)
                return;
            total_comm_bytes_.fetch_add(bytes, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(mu_comm_);
            comm_stats_[(int)op].record(bytes);
        }

        // Scoped helper for timing a collective
        class CommScope
        {
        public:
            CommScope(CommOp op, uint64_t bytes) : op_(op), bytes_(bytes), active_(debugEnv().performance.enable)
            {
                if (active_)
                    start_ = std::chrono::high_resolution_clock::now();
            }
            ~CommScope()
            {
                if (active_)
                {
                    auto end = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
                    PerformanceCounters::instance().record_comm_time(op_, bytes_, ms);
                }
            }

        private:
            CommOp op_;
            uint64_t bytes_;
            bool active_;
            std::chrono::high_resolution_clock::time_point start_{};
        };

        void record_comm_time(CommOp op, uint64_t bytes, double ms)
        {
            {
                std::lock_guard<std::mutex> lk(mu_time_);
                total_comm_time_ms_ += ms;
            }
            record_comm(op, bytes); // reuse path
        }

        struct Snapshot
        {
            uint64_t total_matmuls = 0;
            uint64_t total_flops = 0;
            double total_time_ms = 0;
            uint64_t total_comm_bytes = 0;
            double total_comm_time_ms = 0;
            std::vector<MatmulOpSample> samples;
            std::vector<CommOpStats> comm;
            std::vector<BackendAggStats> backends;
        };
        Snapshot snapshot()
        {
            Snapshot s;
            s.total_matmuls = total_matmuls_.load();
            s.total_flops = total_flops_.load();
            {
                std::lock_guard<std::mutex> lk(mu_time_);
                s.total_time_ms = total_time_ms_;
                s.total_comm_time_ms = total_comm_time_ms_;
                if (has_backend_data())
                {
                    for (int i = 0; i < kMaxBackends; ++i)
                    {
                        if (backend_counts_[i] == 0)
                            continue;
                        BackendAggStats b;
                        b.backend = i;
                        b.count = backend_counts_[i];
                        b.flops = backend_flops_[i];
                        b.time_ms = backend_time_ms_[i];
                        b.min_ms = (backend_min_ms_[i] == std::numeric_limits<double>::max() ? 0.0 : backend_min_ms_[i]);
                        b.max_ms = backend_max_ms_[i];
                        b.prefill_count = backend_prefill_counts_[i];
                        b.decode_count = backend_decode_counts_[i];
                        b.prefill_flops = backend_prefill_flops_[i];
                        b.decode_flops = backend_decode_flops_[i];
                        b.prefill_time_ms = backend_prefill_time_ms_[i];
                        b.decode_time_ms = backend_decode_time_ms_[i];
                        b.prefill_min_ms = (backend_prefill_min_ms_[i] == std::numeric_limits<double>::max() ? 0.0 : backend_prefill_min_ms_[i]);
                        b.prefill_max_ms = backend_prefill_max_ms_[i];
                        b.decode_min_ms = (backend_decode_min_ms_[i] == std::numeric_limits<double>::max() ? 0.0 : backend_decode_min_ms_[i]);
                        b.decode_max_ms = backend_decode_max_ms_[i];
                        s.backends.push_back(b);
                    }
                }
            }
            s.total_comm_bytes = total_comm_bytes_.load();
            {
                std::lock_guard<std::mutex> lk(mu_comm_);
                s.comm.assign(std::begin(comm_stats_), std::end(comm_stats_));
            }
            if (debugEnv().performance.log_each_matmul)
            {
                std::lock_guard<std::mutex> lk(mu_);
                s.samples = samples_;
            }
            return s;
        }

        void set_rank(int r) { rank_cached_ = r; }

        static const char *backendName(int id)
        {
            switch (id)
            {
            case 0:
                return "OPENBLAS"; // MatMulBackend::OPENBLAS
            case 1:
                return "COSMA"; // MatMulBackend::COSMA
            default:
                return "UNKNOWN";
            }
        }
        static const char *layerTypeName(uint8_t lt)
        {
            switch (lt)
            {
            case 1:
                return "MLP";
            case 2:
                return "ATTN";
            default:
                return "UNK";
            }
        }
        static const char *mlpPhaseName(uint8_t ph)
        {
            switch (ph)
            {
            case 1:
                return "gate";
            case 2:
                return "up";
            case 3:
                return "down";
            case 4:
                return "parity_ref";
            default:
                return "other";
            }
        }
        static const char *attnPhaseName(uint8_t ph)
        {
            switch (ph)
            {
            case 1:
                return "q_proj";
            case 2:
                return "k_proj";
            case 3:
                return "v_proj";
            case 4:
                return "out_proj";
            default:
                return "other";
            }
        }

    private:
        PerformanceCounters()
        {
            for (int i = 0; i < kMaxBackends; ++i)
            {
                backend_min_ms_[i] = std::numeric_limits<double>::max();
                backend_prefill_min_ms_[i] = std::numeric_limits<double>::max();
                backend_decode_min_ms_[i] = std::numeric_limits<double>::max();
            }
        }
        static constexpr int kMaxBackends = 8; // adjust if new backends added
        bool has_backend_data() const
        {
            for (int i = 0; i < kMaxBackends; ++i)
                if (backend_counts_[i] > 0)
                    return true;
            return false;
        }
        std::atomic<uint64_t> total_matmuls_{0};
        std::atomic<uint64_t> total_flops_{0};
        std::atomic<uint64_t> total_comm_bytes_{0};
        double total_comm_time_ms_ = 0.0; // guarded by mu_time_
        double total_time_ms_ = 0.0;      // guarded by mu_time_
        std::vector<MatmulOpSample> samples_;
        std::mutex mu_;
        std::mutex mu_time_;
        std::mutex mu_comm_;
        CommOpStats comm_stats_[9];
        // Backend aggregates (protected by mu_time_)
        uint64_t backend_counts_[kMaxBackends] = {0};
        uint64_t backend_flops_[kMaxBackends] = {0};
        double backend_time_ms_[kMaxBackends] = {0.0};
        double backend_min_ms_[kMaxBackends];
        double backend_max_ms_[kMaxBackends] = {0.0};
        // Phase-specific aggregates
        uint64_t backend_prefill_counts_[kMaxBackends] = {0};
        uint64_t backend_decode_counts_[kMaxBackends] = {0};
        uint64_t backend_prefill_flops_[kMaxBackends] = {0};
        uint64_t backend_decode_flops_[kMaxBackends] = {0};
        double backend_prefill_time_ms_[kMaxBackends] = {0.0};
        double backend_decode_time_ms_[kMaxBackends] = {0.0};
        double backend_prefill_min_ms_[kMaxBackends];
        double backend_prefill_max_ms_[kMaxBackends] = {0.0};
        double backend_decode_min_ms_[kMaxBackends];
        double backend_decode_max_ms_[kMaxBackends] = {0.0};
        int rank_cached_ = 0; // set by caller early
    };

    inline PerformanceCounters &perfCounters() { return PerformanceCounters::instance(); }
    inline void setPerfMatmulPhase(uint8_t layer_type, uint8_t phase, int32_t layer_index = -1) { PerformanceCounters::setPhase(layer_type, phase, layer_index); }
    struct PerfMatmulPhaseScope
    {
        PerformanceCounters::PhaseScope scope;
        PerfMatmulPhaseScope(uint8_t lt, uint8_t ph, int32_t idx = -1) : scope(lt, ph, idx) {}
    };

    // Definition of thread-local phase context
    inline thread_local PerformanceCounters::PhaseContext PerformanceCounters::tl_phase_{};

// Lightweight wrappers (inline) for instrumentation; only active if LLAMINAR_PERF_ENABLE.
#include <mpi.h>
    inline int PerfAllreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    {
        int dtype_size = 0;
        MPI_Type_size(datatype, &dtype_size);
        uint64_t bytes = (uint64_t)count * (uint64_t)dtype_size;
        PerformanceCounters::CommScope scope(CommOp::Allreduce, bytes);
        return MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }
    inline int PerfAllgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
    {
        int st = 0;
        MPI_Type_size(sendtype, &st);
        uint64_t bytes = (uint64_t)sendcount * (uint64_t)st;
        PerformanceCounters::CommScope scope(CommOp::Allgather, bytes);
        return MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }
    inline int PerfAllgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)
    {
        int st = 0;
        MPI_Type_size(sendtype, &st);
        uint64_t bytes = (uint64_t)sendcount * (uint64_t)st;
        PerformanceCounters::CommScope scope(CommOp::Allgatherv, bytes);
        return MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    }
    inline int PerfBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    {
        int st = 0;
        MPI_Type_size(datatype, &st);
        uint64_t bytes = (uint64_t)count * (uint64_t)st;
        PerformanceCounters::CommScope scope(CommOp::Bcast, bytes);
        return MPI_Bcast(buffer, count, datatype, root, comm);
    }
    inline int PerfReduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
    {
        int st = 0;
        MPI_Type_size(datatype, &st);
        uint64_t bytes = (uint64_t)count * (uint64_t)st;
        PerformanceCounters::CommScope scope(CommOp::Reduce, bytes);
        return MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    }
    inline int PerfReduceScatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    {
        int st = 0;
        MPI_Type_size(datatype, &st);
        int world = 0;
        MPI_Comm_size(comm, &world);
        uint64_t total = 0;
        for (int i = 0; i < world; ++i)
            total += (uint64_t)recvcounts[i];
        uint64_t bytes = total * (uint64_t)st;
        PerformanceCounters::CommScope scope(CommOp::ReduceScatter, bytes);
        return MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
    }
    inline int PerfAlltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
    {
        int st = 0;
        MPI_Type_size(sendtype, &st);
        uint64_t bytes = (uint64_t)sendcount * (uint64_t)st;
        PerformanceCounters::CommScope scope(CommOp::Alltoall, bytes);
        return MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }
    inline int PerfAlltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)
    {
        int st = 0;
        MPI_Type_size(sendtype, &st);
        int world = 0;
        MPI_Comm_size(comm, &world);
        uint64_t total = 0;
        for (int i = 0; i < world; ++i)
            total += (uint64_t)sendcounts[i];
        uint64_t bytes = total * (uint64_t)st;
        PerformanceCounters::CommScope scope(CommOp::Alltoallv, bytes);
        return MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
    }
    inline int PerfBarrier(MPI_Comm comm)
    {
        PerformanceCounters::CommScope scope(CommOp::Barrier, 0);
        return MPI_Barrier(comm);
    }

} // namespace llaminar
