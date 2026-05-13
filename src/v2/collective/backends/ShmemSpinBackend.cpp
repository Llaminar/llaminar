/**
 * @file ShmemSpinBackend.cpp
 * @brief Shared-memory spin-wait collective backend implementation
 *
 * Fast-path allreduce for N-rank intra-node CPU tensor parallelism.
 * Uses POSIX shared memory + atomic epoch counters + AVX-512 reduction.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include "ShmemSpinBackend.h"
#include "../../utils/Assertions.h"
#include "../../utils/CPUFeatures.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"

#include <chrono>
#include <cstring>    // memcpy
#include <fcntl.h>    // O_CREAT, O_RDWR
#include <immintrin.h>
#include <limits>
#include <mpi.h>
#include <sys/mman.h> // shm_open, mmap, munmap, shm_unlink
#include <unistd.h>   // ftruncate, close

namespace llaminar2
{
    namespace
    {
        constexpr uint64_t kAbortEpoch = std::numeric_limits<uint64_t>::max();
        constexpr int kDefaultReleaseTimeoutMs = 300000;

        int shmemSpinTimeoutMs()
        {
            const int configured = debugEnv().tp_collect_timeout_ms;
            if (configured < 0)
                return 0;
            return configured > 0 ? configured : kDefaultReleaseTimeoutMs;
        }
    } // namespace


    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    ShmemSpinBackend::ShmemSpinBackend(int domain_id, int my_rank,
                                       std::unique_ptr<UPICollectiveBackend> fallback)
        : domain_id_(domain_id), my_rank_(my_rank), fallback_(std::move(fallback))
    {
        // Build deterministic shm name from domain ID
        shm_name_ = "/llaminar_shmem_ar_" + std::to_string(domain_id_);

        LOG_DEBUG("ShmemSpinBackend: Created for domain " << domain_id_
                                                          << " rank " << my_rank_
                                                          << " (shm=" << shm_name_ << ")");
    }

    ShmemSpinBackend::~ShmemSpinBackend()
    {
        shutdown();
    }

    // =========================================================================
    // Capability Queries
    // =========================================================================

    bool ShmemSpinBackend::supportsDevice(DeviceType type) const
    {
        return type == DeviceType::CPU;
    }

    bool ShmemSpinBackend::supportsDirectTransfer(DeviceId src, DeviceId dst) const
    {
        return src.type == DeviceType::CPU && dst.type == DeviceType::CPU;
    }

    bool ShmemSpinBackend::isAvailable() const
    {
        return arena_ != nullptr && fallback_ && fallback_->isAvailable();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool ShmemSpinBackend::initialize(const DeviceGroup &group)
    {
        if (initialized_)
        {
            return true;
        }

        // Derive rank count from device group
        num_ranks_ = static_cast<int>(group.devices.size());
        if (num_ranks_ < 1)
        {
            last_error_ = "Device group must have at least 1 device";
            LOG_ERROR("ShmemSpinBackend::initialize - " << last_error_);
            return false;
        }

        if (my_rank_ >= num_ranks_)
        {
            last_error_ = "my_rank (" + std::to_string(my_rank_) + ") >= num_ranks (" + std::to_string(num_ranks_) + ")";
            LOG_ERROR("ShmemSpinBackend::initialize - " << last_error_);
            return false;
        }

        // Initialize fallback backend FIRST — setupSharedMemory() uses
        // fallback_->synchronize() for inter-rank barriers.
        if (fallback_ && !fallback_->isInitialized())
        {
            if (!fallback_->initialize(group))
            {
                last_error_ = "Failed to initialize fallback UPI backend";
                LOG_ERROR("ShmemSpinBackend::initialize - " << last_error_);
                return false;
            }
        }

        // Set up shared memory (requires working fallback for barriers)
        if (!setupSharedMemory())
        {
            LOG_ERROR("ShmemSpinBackend::initialize - Failed to set up shared memory");
            return false;
        }

        abort_requested_.store(false, std::memory_order_release);
        initialized_ = true;

        LOG_DEBUG("ShmemSpinBackend initialized: domain=" << domain_id_
                                                          << " rank=" << my_rank_
                                                          << " num_ranks=" << num_ranks_
                                                          << " max_count=" << ShmemSpinArena::MAX_COUNT
                                                          << " arena_bytes=" << arena_size_);
        return true;
    }

    bool ShmemSpinBackend::isInitialized() const
    {
        return initialized_;
    }

    void ShmemSpinBackend::shutdown()
    {
        if (!initialized_)
        {
            return;
        }

        initialized_ = false;
        teardownSharedMemory();

        if (fallback_)
        {
            fallback_->shutdown();
        }

        LOG_DEBUG("ShmemSpinBackend shutdown: domain=" << domain_id_ << " rank=" << my_rank_);
    }

    void ShmemSpinBackend::abort()
    {
        abort_requested_.store(true, std::memory_order_release);
        if (arena_ && my_rank_ >= 0)
            arena_->epoch_at(my_rank_)->epoch.store(kAbortEpoch, std::memory_order_release);
        if (fallback_)
            fallback_->abort();
    }

    // =========================================================================
    // Shared Memory Setup / Teardown
    // =========================================================================

    bool ShmemSpinBackend::setupSharedMemory()
    {
        // All ranks create-or-open the same named segment.
        // Rank 0 creates+truncates; other ranks open existing.
        // MPI barrier (via fallback) ensures ordering.

        arena_size_ = ShmemSpinArena::compute_size(num_ranks_);

        if (my_rank_ == 0)
        {
            // Rank 0: create the segment (unlink first to ensure clean state)
            shm_unlink(shm_name_.c_str()); // Ignore errors (may not exist)

            shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0600);
            if (shm_fd_ < 0)
            {
                last_error_ = "shm_open(create) failed: " + std::string(strerror(errno));
                LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
                return false;
            }

            if (ftruncate(shm_fd_, static_cast<off_t>(arena_size_)) != 0)
            {
                last_error_ = "ftruncate failed: " + std::string(strerror(errno));
                LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
                close(shm_fd_);
                shm_fd_ = -1;
                shm_unlink(shm_name_.c_str());
                return false;
            }
        }

        // Barrier: rank 0 finishes creation before others open
        if (fallback_)
        {
            fallback_->synchronize();
        }

        if (my_rank_ != 0)
        {
            // Non-zero ranks: open existing segment
            shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0600);
            if (shm_fd_ < 0)
            {
                last_error_ = "shm_open(open) failed: " + std::string(strerror(errno));
                LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
                return false;
            }
        }

        // Map the shared memory
        void *ptr = mmap(nullptr, arena_size_,
                         PROT_READ | PROT_WRITE, MAP_SHARED,
                         shm_fd_, 0);

        if (ptr == MAP_FAILED)
        {
            last_error_ = "mmap failed: " + std::string(strerror(errno));
            LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }

        arena_ = static_cast<ShmemSpinArena *>(ptr);

        // Rank 0 zero-initializes the arena and writes the header
        if (my_rank_ == 0)
        {
            std::memset(arena_, 0, arena_size_);
            arena_->num_ranks = num_ranks_;
        }

        // Barrier: ensure zero-init completes before anyone uses it
        if (fallback_)
        {
            fallback_->synchronize();
        }

        LOG_DEBUG("ShmemSpinBackend: Shared memory mapped at " << ptr
                                                               << " (" << arena_size_ << " bytes)"
                                                               << " for rank " << my_rank_
                                                               << " of " << num_ranks_);
        return true;
    }

    void ShmemSpinBackend::teardownSharedMemory()
    {
        if (arena_)
        {
            munmap(arena_, arena_size_);
            arena_ = nullptr;
        }

        if (shm_fd_ >= 0)
        {
            close(shm_fd_);
            shm_fd_ = -1;
        }

        // Rank 0 unlinks the segment (safe even if rank 1 still has it mapped)
        if (my_rank_ == 0)
        {
            shm_unlink(shm_name_.c_str());
        }
    }

    // =========================================================================
    // Fast-Path Check
    // =========================================================================

    bool ShmemSpinBackend::isFastPath(size_t count, CollectiveDataType dtype,
                                      CollectiveOp op) const
    {
        if (op != CollectiveOp::ALLREDUCE_SUM || count > ShmemSpinArena::MAX_COUNT)
            return false;

        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
        case CollectiveDataType::FLOAT16:
        case CollectiveDataType::BFLOAT16:
            return true;
        default:
            return false;
        }
    }

    bool ShmemSpinBackend::waitForPeerEpoch(int peer_rank, uint64_t target_epoch, const char *phase, size_t count)
    {
        const int timeout_ms = shmemSpinTimeoutMs();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        uint64_t spins = 0;

        while (true)
        {
            const uint64_t peer_epoch = arena_->epoch_at(peer_rank)->epoch.load(std::memory_order_acquire);
            if (peer_epoch == kAbortEpoch)
            {
                last_error_ = "peer rank " + std::to_string(peer_rank) +
                              " aborted ShmemSpinBackend domain " + std::to_string(domain_id_) +
                              " while " + phase;
                LOG_ERROR("ShmemSpinBackend - " << last_error_);
                abort();
                return false;
            }
            if (peer_epoch >= target_epoch)
                return true;
            if (abort_requested_.load(std::memory_order_acquire))
            {
                last_error_ = "ShmemSpinBackend abort requested while waiting for peer rank " +
                              std::to_string(peer_rank) + " to reach epoch " + std::to_string(target_epoch) +
                              " while " + phase;
                LOG_ERROR("ShmemSpinBackend - " << last_error_);
                abort();
                return false;
            }
            if (timeout_ms > 0 && (++spins & 0x3fffU) == 0 && std::chrono::steady_clock::now() >= deadline)
            {
                last_error_ = "ShmemSpinBackend timed out after " + std::to_string(timeout_ms) +
                              "ms on domain " + std::to_string(domain_id_) +
                              " rank " + std::to_string(my_rank_) +
                              " waiting for peer " + std::to_string(peer_rank) +
                              " while " + phase +
                              " (target_epoch=" + std::to_string(target_epoch) +
                              ", peer_epoch=" + std::to_string(peer_epoch) +
                              ", count=" + std::to_string(count) + ")";
                LOG_ERROR("ShmemSpinBackend - " << last_error_
                          << "; aborting MPI job to avoid rank desynchronization");
                abort();
                MPI_Abort(MPI_COMM_WORLD, 1);
                return false;
            }
            _mm_pause();
        }
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool ShmemSpinBackend::allreduce(void *buffer, size_t count,
                                     CollectiveDataType dtype, CollectiveOp op)
    {
        if (abort_requested_.load(std::memory_order_acquire))
        {
            last_error_ = "ShmemSpinBackend abort has been requested";
            return false;
        }

        if (!initialized_ || !arena_)
        {
            last_error_ = "ShmemSpinBackend not initialized";
            return false;
        }

        // Fast path: SUM with count within shared buffer capacity
        if (isFastPath(count, dtype, op))
        {
            // Element size depends on dtype (isFastPath gates to these three)
            size_t elem_size;
            switch (dtype)
            {
            case CollectiveDataType::FLOAT32:
                elem_size = sizeof(float);
                break;
            case CollectiveDataType::FLOAT16:
            case CollectiveDataType::BFLOAT16:
                elem_size = sizeof(uint16_t);
                break;
            default:
                LLAMINAR_UNREACHABLE("isFastPath passed unsupported dtype");
            }

            // 1. Stage my data into shared memory
            std::memcpy(arena_->buffer_at(my_rank_), buffer, count * elem_size);

            // 2. Increment and signal ready (store-release)
            my_epoch_++;
            arena_->epoch_at(my_rank_)->epoch.store(my_epoch_, std::memory_order_release);

            // 3. Spin-wait for ALL peers (load-acquire)
            for (int r = 0; r < num_ranks_; ++r)
            {
                if (r == my_rank_)
                    continue;
                if (!waitForPeerEpoch(r, my_epoch_, "entering allreduce fast path", count))
                    return false;
            }

            // 4. N-way reduce: accumulate all rank buffers
            if (num_ranks_ == 1)
            {
                // Trivial: just copy our own buffer back
                std::memcpy(buffer, arena_->buffer_at(0), count * elem_size);
            }
            else
            {
                switch (dtype)
                {
                case CollectiveDataType::FLOAT32:
                {
                    auto *out = static_cast<float *>(buffer);
                    reduce(out, arena_->buffer_at(0), arena_->buffer_at(1), count);
                    for (int r = 2; r < num_ranks_; ++r)
                        reduce(out, out, arena_->buffer_at(r), count);
                    break;
                }
                case CollectiveDataType::FLOAT16:
                {
                    auto *out = static_cast<uint16_t *>(buffer);
                    const auto *buf0 = reinterpret_cast<const uint16_t *>(arena_->buffer_at(0));
                    const auto *buf1 = reinterpret_cast<const uint16_t *>(arena_->buffer_at(1));
                    reduce_fp16(out, buf0, buf1, count);
                    for (int r = 2; r < num_ranks_; ++r)
                    {
                        const auto *bufr = reinterpret_cast<const uint16_t *>(arena_->buffer_at(r));
                        reduce_fp16(out, out, bufr, count);
                    }
                    break;
                }
                case CollectiveDataType::BFLOAT16:
                {
                    auto *out = static_cast<uint16_t *>(buffer);
                    const auto *buf0 = reinterpret_cast<const uint16_t *>(arena_->buffer_at(0));
                    const auto *buf1 = reinterpret_cast<const uint16_t *>(arena_->buffer_at(1));
                    reduce_bf16(out, buf0, buf1, count);
                    for (int r = 2; r < num_ranks_; ++r)
                    {
                        const auto *bufr = reinterpret_cast<const uint16_t *>(arena_->buffer_at(r));
                        reduce_bf16(out, out, bufr, count);
                    }
                    break;
                }
                default:
                    break; // unreachable — isFastPath guards this
                }
            }

            // 5. Read-completion barrier: ensure all ranks finished reducing
            //    before any rank re-enters and overwrites arena buffers.
            my_epoch_++;
            arena_->epoch_at(my_rank_)->epoch.store(my_epoch_, std::memory_order_release);
            for (int r = 0; r < num_ranks_; ++r)
            {
                if (r == my_rank_)
                    continue;
                if (!waitForPeerEpoch(r, my_epoch_, "leaving allreduce fast path", count))
                    return false;
            }

            return true;
        }

        // Fallback to MPI for non-fast-path operations
        if (fallback_)
        {
            return fallback_->allreduce(buffer, count, dtype, op);
        }

        last_error_ = "No fallback backend for non-fast-path allreduce";
        return false;
    }

    bool ShmemSpinBackend::allgather(const void *send_buf, void *recv_buf,
                                     size_t send_count, CollectiveDataType dtype)
    {
        if (fallback_)
        {
            return fallback_->allgather(send_buf, recv_buf, send_count, dtype);
        }
        last_error_ = "No fallback backend for allgather";
        return false;
    }

    bool ShmemSpinBackend::allgatherv(const void *send_buf, size_t send_count,
                                      void *recv_buf,
                                      const std::vector<int> &recv_counts,
                                      const std::vector<int> &displacements,
                                      CollectiveDataType dtype)
    {
        if (fallback_)
        {
            return fallback_->allgatherv(send_buf, send_count, recv_buf,
                                         recv_counts, displacements, dtype);
        }
        last_error_ = "No fallback backend for allgatherv";
        return false;
    }

    bool ShmemSpinBackend::reduceScatter(const void *send_buf, void *recv_buf,
                                         size_t recv_count, CollectiveDataType dtype,
                                         CollectiveOp op)
    {
        if (fallback_)
        {
            return fallback_->reduceScatter(send_buf, recv_buf, recv_count, dtype, op);
        }
        last_error_ = "No fallback backend for reduceScatter";
        return false;
    }

    bool ShmemSpinBackend::broadcast(void *buffer, size_t count,
                                     CollectiveDataType dtype, int root_rank)
    {
        if (fallback_)
        {
            return fallback_->broadcast(buffer, count, dtype, root_rank);
        }
        last_error_ = "No fallback backend for broadcast";
        return false;
    }

    bool ShmemSpinBackend::synchronize()
    {
        if (fallback_)
        {
            return fallback_->synchronize();
        }
        // Without fallback, use epoch-based N-way barrier
        if (!arena_)
        {
            return false;
        }
        my_epoch_++;
        arena_->epoch_at(my_rank_)->epoch.store(my_epoch_, std::memory_order_release);
        for (int r = 0; r < num_ranks_; ++r)
        {
            if (r == my_rank_)
                continue;
            if (!waitForPeerEpoch(r, my_epoch_, "synchronizing", 0))
                return false;
        }
        return true;
    }

    // =========================================================================
    // Vectorized Reduction — three ISA paths + runtime dispatch
    // =========================================================================

    // ---- FP16/BF16 conversion helpers (file-local) -------------------------

    namespace
    {
        // BF16 ↔ FP32: upper 16 bits of IEEE-754 float
        inline float bf16_to_float(uint16_t bf)
        {
            uint32_t f = static_cast<uint32_t>(bf) << 16;
            float result;
            std::memcpy(&result, &f, sizeof(float));
            return result;
        }

        inline uint16_t float_to_bf16(float val)
        {
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(float));
            return static_cast<uint16_t>(bits >> 16); // truncation
        }

        // FP16 ↔ FP32: IEEE-754 half-precision (software, no F16C required)
        inline float fp16_to_float(uint16_t h)
        {
            uint32_t sign = static_cast<uint32_t>(h >> 15) << 31;
            uint32_t exp = (h >> 10) & 0x1Fu;
            uint32_t mant = h & 0x3FFu;
            uint32_t f;

            if (exp == 0)
            {
                if (mant == 0)
                {
                    f = sign; // ±0
                }
                else
                {
                    // Subnormal: normalize
                    exp = 1;
                    while (!(mant & 0x400u))
                    {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x3FFu;
                    f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
                }
            }
            else if (exp == 31)
            {
                f = sign | 0x7F800000u | (mant << 13); // Inf/NaN
            }
            else
            {
                f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
            }

            float result;
            std::memcpy(&result, &f, sizeof(float));
            return result;
        }

        inline uint16_t float_to_fp16(float val)
        {
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(float));
            uint32_t sign = (bits >> 16) & 0x8000u;
            int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
            uint32_t mant = bits & 0x7FFFFFu;

            if (exp <= 0)
            {
                if (exp < -10)
                    return static_cast<uint16_t>(sign); // ±0
                // Subnormal
                mant |= 0x800000u;
                uint32_t shift = static_cast<uint32_t>(1 - exp + 13);
                uint32_t round_bit = 1u << (shift - 1);
                uint32_t remainder = mant & ((1u << shift) - 1);
                mant >>= shift;
                if (remainder > round_bit || (remainder == round_bit && (mant & 1)))
                    mant++;
                return static_cast<uint16_t>(sign | mant);
            }
            else if (exp >= 31)
            {
                if (exp == (0xFF - 127 + 15) && mant)
                    return static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13)); // NaN
                return static_cast<uint16_t>(sign | 0x7C00u);                    // Inf
            }

            // Round to nearest even
            uint32_t round_bit = 1u << 12;
            uint32_t remainder = mant & 0x1FFFu;
            mant >>= 13;
            if (remainder > round_bit || (remainder == round_bit && (mant & 1)))
                mant++;
            if (mant >= 0x400u)
            {
                mant = 0;
                exp++;
                if (exp >= 31)
                    return static_cast<uint16_t>(sign | 0x7C00u);
            }
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
        }
    } // anonymous namespace

    // ---- FP32 reduce (existing) --------------------------------------------

    void ShmemSpinBackend::reduce_scalar(float *out,
                                         const float *a,
                                         const float *b,
                                         size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            out[i] = a[i] + b[i];
        }
    }

    void ShmemSpinBackend::reduce_avx2(float *out,
                                       const float *a,
                                       const float *b,
                                       size_t count)
    {
#if defined(__AVX2__)
        size_t i = 0;
        const size_t vec_end = count & ~size_t(7); // Round down to multiple of 8
        for (; i < vec_end; i += 8)
        {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
        }
        // Scalar tail
        for (; i < count; ++i)
        {
            out[i] = a[i] + b[i];
        }
#else
        reduce_scalar(out, a, b, count);
#endif
    }

    void ShmemSpinBackend::reduce_avx512(float *out,
                                          const float *a,
                                          const float *b,
                                          size_t count)
    {
#if defined(__AVX512F__)
        size_t i = 0;
        const size_t vec_end = count & ~size_t(15); // Round down to multiple of 16
        for (; i < vec_end; i += 16)
        {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            _mm512_storeu_ps(out + i, _mm512_add_ps(va, vb));
        }
        // Masked tail (0-15 remaining elements)
        if (i < count)
        {
            const __mmask16 mask = (__mmask16)((1u << (count - i)) - 1);
            __m512 va = _mm512_maskz_loadu_ps(mask, a + i);
            __m512 vb = _mm512_maskz_loadu_ps(mask, b + i);
            _mm512_mask_storeu_ps(out + i, mask, _mm512_add_ps(va, vb));
        }
#else
        reduce_avx2(out, a, b, count);
#endif
    }

    void ShmemSpinBackend::reduce(float *out,
                                  const float *a,
                                  const float *b,
                                  size_t count)
    {
        switch (activeISALevel())
        {
        case ISALevel::AVX512:
            reduce_avx512(out, a, b, count);
            break;
        case ISALevel::AVX2:
            reduce_avx2(out, a, b, count);
            break;
        default:
            reduce_scalar(out, a, b, count);
            break;
        }
    }

    // ---- FP16 reduce -------------------------------------------------------

    void ShmemSpinBackend::reduce_fp16_scalar(uint16_t *out,
                                              const uint16_t *a,
                                              const uint16_t *b,
                                              size_t count)
    {
#if defined(__F16C__)
        for (size_t i = 0; i < count; ++i)
        {
            float fa = _cvtsh_ss(a[i]);
            float fb = _cvtsh_ss(b[i]);
            out[i] = static_cast<uint16_t>(
                _cvtss_sh(fa + fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
#else
        for (size_t i = 0; i < count; ++i)
        {
            out[i] = float_to_fp16(fp16_to_float(a[i]) + fp16_to_float(b[i]));
        }
#endif
    }

    void ShmemSpinBackend::reduce_fp16_avx2(uint16_t *out,
                                            const uint16_t *a,
                                            const uint16_t *b,
                                            size_t count)
    {
#if defined(__AVX2__) && defined(__F16C__)
        size_t i = 0;
        const size_t vec_end = count & ~size_t(7); // 8 FP16 at a time
        for (; i < vec_end; i += 8)
        {
            __m128i ha = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
            __m128i hb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
            __m256 fa = _mm256_cvtph_ps(ha);
            __m256 fb = _mm256_cvtph_ps(hb);
            __m128i result = _mm256_cvtps_ph(_mm256_add_ps(fa, fb),
                                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), result);
        }
        // Scalar tail (F16C available since AVX2 implies it)
        for (; i < count; ++i)
        {
            float fa = _cvtsh_ss(a[i]);
            float fb = _cvtsh_ss(b[i]);
            out[i] = static_cast<uint16_t>(
                _cvtss_sh(fa + fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
#else
        reduce_fp16_scalar(out, a, b, count);
#endif
    }

    void ShmemSpinBackend::reduce_fp16_avx512(uint16_t *out,
                                              const uint16_t *a,
                                              const uint16_t *b,
                                              size_t count)
    {
#if defined(__AVX512F__)
        size_t i = 0;
        const size_t vec_end = count & ~size_t(15); // 16 FP16 at a time
        for (; i < vec_end; i += 16)
        {
            __m256i ha = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
            __m256i hb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
            __m512 fa = _mm512_cvtph_ps(ha);
            __m512 fb = _mm512_cvtph_ps(hb);
            __m256i result = _mm512_cvtps_ph(_mm512_add_ps(fa, fb),
                                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), result);
        }
        // F16C scalar tail
        for (; i < count; ++i)
        {
            float fa = _cvtsh_ss(a[i]);
            float fb = _cvtsh_ss(b[i]);
            out[i] = static_cast<uint16_t>(
                _cvtss_sh(fa + fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
#else
        reduce_fp16_avx2(out, a, b, count);
#endif
    }

    void ShmemSpinBackend::reduce_fp16(uint16_t *out,
                                       const uint16_t *a,
                                       const uint16_t *b,
                                       size_t count)
    {
        switch (activeISALevel())
        {
        case ISALevel::AVX512:
            reduce_fp16_avx512(out, a, b, count);
            break;
        case ISALevel::AVX2:
            reduce_fp16_avx2(out, a, b, count);
            break;
        default:
            reduce_fp16_scalar(out, a, b, count);
            break;
        }
    }

    // ---- BF16 reduce (bit-shift, no native BF16 ops) -----------------------

    void ShmemSpinBackend::reduce_bf16_scalar(uint16_t *out,
                                              const uint16_t *a,
                                              const uint16_t *b,
                                              size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            out[i] = float_to_bf16(bf16_to_float(a[i]) + bf16_to_float(b[i]));
        }
    }

    void ShmemSpinBackend::reduce_bf16_avx2(uint16_t *out,
                                            const uint16_t *a,
                                            const uint16_t *b,
                                            size_t count)
    {
#if defined(__AVX2__)
        size_t i = 0;
        const size_t vec_end = count & ~size_t(7); // 8 BF16 at a time
        for (; i < vec_end; i += 8)
        {
            // Load 8 × uint16 → zero-extend to 8 × uint32
            __m128i raw_a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
            __m128i raw_b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
            __m256i a32 = _mm256_cvtepu16_epi32(raw_a);
            __m256i b32 = _mm256_cvtepu16_epi32(raw_b);

            // BF16 → FP32: shift left 16
            __m256 fa = _mm256_castsi256_ps(_mm256_slli_epi32(a32, 16));
            __m256 fb = _mm256_castsi256_ps(_mm256_slli_epi32(b32, 16));
            __m256 sum = _mm256_add_ps(fa, fb);

            // FP32 → BF16: shift right 16 (truncation)
            __m256i sum_i = _mm256_srli_epi32(_mm256_castps_si256(sum), 16);

            // Pack 8 × uint32 → 8 × uint16 via SSE4.1 packus
            __m128i lo = _mm256_castsi256_si128(sum_i);
            __m128i hi = _mm256_extracti128_si256(sum_i, 1);
            __m128i packed = _mm_packus_epi32(lo, hi);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), packed);
        }
        // Scalar tail
        for (; i < count; ++i)
        {
            out[i] = float_to_bf16(bf16_to_float(a[i]) + bf16_to_float(b[i]));
        }
#else
        reduce_bf16_scalar(out, a, b, count);
#endif
    }

    void ShmemSpinBackend::reduce_bf16_avx512(uint16_t *out,
                                              const uint16_t *a,
                                              const uint16_t *b,
                                              size_t count)
    {
#if defined(__AVX512F__)
        size_t i = 0;
        const size_t vec_end = count & ~size_t(15); // 16 BF16 at a time
        for (; i < vec_end; i += 16)
        {
            // Load 16 × uint16 → zero-extend to 16 × uint32
            __m256i raw_a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
            __m256i raw_b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
            __m512i a32 = _mm512_cvtepu16_epi32(raw_a);
            __m512i b32 = _mm512_cvtepu16_epi32(raw_b);

            // BF16 → FP32: shift left 16
            __m512 fa = _mm512_castsi512_ps(_mm512_slli_epi32(a32, 16));
            __m512 fb = _mm512_castsi512_ps(_mm512_slli_epi32(b32, 16));
            __m512 sum = _mm512_add_ps(fa, fb);

            // FP32 → BF16: shift right 16 (truncation)
            __m512i sum_i = _mm512_srli_epi32(_mm512_castps_si512(sum), 16);

            // Pack 16 × uint32 → 16 × uint16
            // Extract 256-bit halves, use SSE4.1 packus per half
            __m256i lo256 = _mm512_castsi512_si256(sum_i);
            __m256i hi256 = _mm512_extracti64x4_epi64(sum_i, 1);
            __m128i a128 = _mm256_castsi256_si128(lo256);
            __m128i b128 = _mm256_extracti128_si256(lo256, 1);
            __m128i c128 = _mm256_castsi256_si128(hi256);
            __m128i d128 = _mm256_extracti128_si256(hi256, 1);
            __m256i packed = _mm256_set_m128i(_mm_packus_epi32(c128, d128),
                                             _mm_packus_epi32(a128, b128));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), packed);
        }
        // Scalar tail
        for (; i < count; ++i)
        {
            out[i] = float_to_bf16(bf16_to_float(a[i]) + bf16_to_float(b[i]));
        }
#else
        reduce_bf16_avx2(out, a, b, count);
#endif
    }

    void ShmemSpinBackend::reduce_bf16(uint16_t *out,
                                       const uint16_t *a,
                                       const uint16_t *b,
                                       size_t count)
    {
        switch (activeISALevel())
        {
        case ISALevel::AVX512:
            reduce_bf16_avx512(out, a, b, count);
            break;
        case ISALevel::AVX2:
            reduce_bf16_avx2(out, a, b, count);
            break;
        default:
            reduce_bf16_scalar(out, a, b, count);
            break;
        }
    }

} // namespace llaminar2
