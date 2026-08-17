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
#include "../CollectiveTimeoutPolicy.h"
#include "../../utils/Assertions.h"
#include "../../utils/CPUFeatures.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>    // memcpy
#include <fcntl.h>    // O_CREAT, O_RDWR
#include <immintrin.h>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <sstream>
#include <sys/mman.h> // shm_open, mmap, munmap, shm_unlink
#include <unistd.h>   // ftruncate, close

namespace llaminar2
{
    namespace
    {
        constexpr uint64_t kAbortEpoch = std::numeric_limits<uint64_t>::max();
        constexpr int kMaxCreateAttempts = 64;

        std::atomic<uint64_t> g_shmem_name_counter{0};

        int shmemSpinTimeoutMs()
        {
            return collective_timeout_policy::effectiveCollectTimeoutMs(
                debugEnv().tp_collect_timeout_ms);
        }

        std::string makeUniqueShmemName(int domain_id)
        {
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            const uint64_t sequence = g_shmem_name_counter.fetch_add(1, std::memory_order_relaxed) + 1;

            std::ostringstream name;
            name << "/llaminar_shmem_ar_u" << static_cast<unsigned long>(getuid())
                 << "_d" << domain_id
                 << "_p" << static_cast<unsigned long>(getpid())
                 << "_t" << static_cast<unsigned long long>(now)
                 << "_s" << sequence;
            return name.str();
        }

        /** Return the byte width of one collective element. */
        size_t collectiveElementSize(CollectiveDataType dtype)
        {
            switch (dtype)
            {
            case CollectiveDataType::FLOAT32:
            case CollectiveDataType::INT32:
                return 4u;
            case CollectiveDataType::FLOAT16:
            case CollectiveDataType::BFLOAT16:
                return 2u;
            case CollectiveDataType::INT8:
                return 1u;
            }
            return 0u;
        }
    } // namespace


    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    ShmemSpinBackend::ShmemSpinBackend(int domain_id, int my_rank,
                                       std::unique_ptr<UPICollectiveBackend> general_backend)
        : domain_id_(domain_id), my_rank_(my_rank),
          general_backend_(std::move(general_backend))
    {
        LOG_DEBUG("ShmemSpinBackend: Created for domain " << domain_id_
                                                          << " rank " << my_rank_);
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
        return arena_ != nullptr && general_backend_ && general_backend_->isAvailable();
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

        rooted_publication_element_counts_.assign(
            static_cast<size_t>(num_ranks_),
            0u);
        rooted_publication_element_offsets_.assign(
            static_cast<size_t>(num_ranks_),
            0u);

        // Initialize the general backend first because setupSharedMemory() uses
        // its communicator for one-time construction barriers.
        if (general_backend_ && !general_backend_->isInitialized())
        {
            if (!general_backend_->initialize(group))
            {
                last_error_ = "Failed to initialize general UPI backend";
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

        my_epoch_ = 0;
        abort_requested_.store(false, std::memory_order_release);
        initialized_ = true;

        LOG_DEBUG("ShmemSpinBackend initialized: domain=" << domain_id_
                                                          << " rank=" << my_rank_
                                                          << " num_ranks=" << num_ranks_
                                                          << " chunk_capacity=" << ShmemSpinArena::CHUNK_CAPACITY
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

        if (general_backend_)
        {
            general_backend_->shutdown();
        }

        LOG_DEBUG("ShmemSpinBackend shutdown: domain=" << domain_id_ << " rank=" << my_rank_);
    }

    void ShmemSpinBackend::abort()
    {
        abort_requested_.store(true, std::memory_order_release);
        if (arena_ && my_rank_ >= 0)
            arena_->epoch_at(my_rank_)->epoch.store(kAbortEpoch, std::memory_order_release);
        if (general_backend_)
            general_backend_->abort();
    }

    // =========================================================================
    // Shared Memory Setup / Teardown
    // =========================================================================

    bool ShmemSpinBackend::setupSharedMemory()
    {
        // Rank 0 creates a fresh per-run segment with O_EXCL, broadcasts the
        // generated name, then unlinks it after every rank has mapped it. The
        // mapping stays alive through open file descriptors, while the name is
        // removed from /dev/shm so failed or later runs cannot collide with it.

        arena_size_ = ShmemSpinArena::compute_size(num_ranks_);

        if (!general_backend_ || general_backend_->domainComm() == MPI_COMM_NULL)
        {
            last_error_ = "general UPI backend has no domain communicator";
            LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
            return false;
        }

        MPI_Comm comm = general_backend_->domainComm();
        int create_success = 1;
        int name_len = 0;

        if (my_rank_ == 0)
        {
            for (int attempt = 0; attempt < kMaxCreateAttempts; ++attempt)
            {
                shm_name_ = makeUniqueShmemName(domain_id_);
                shm_unlinked_ = false;

                shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
                if (shm_fd_ >= 0)
                    break;

                if (errno == EEXIST)
                    continue;

                last_error_ = "shm_open(create) failed for " + shm_name_ + ": " + std::string(strerror(errno));
                break;
            }

            if (shm_fd_ < 0)
            {
                if (last_error_.empty())
                    last_error_ = "shm_open(create) exhausted unique-name attempts";
                LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
                create_success = 0;
            }
            else if (ftruncate(shm_fd_, static_cast<off_t>(arena_size_)) != 0)
            {
                last_error_ = "ftruncate failed for " + shm_name_ + ": " + std::string(strerror(errno));
                LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
                close(shm_fd_);
                shm_fd_ = -1;
                shm_unlink(shm_name_.c_str());
                shm_unlinked_ = true;
                create_success = 0;
            }

            name_len = static_cast<int>(shm_name_.size());
        }

        MPI_Bcast(&create_success, 1, MPI_INT, 0, comm);
        if (!create_success)
        {
            if (my_rank_ != 0)
                last_error_ = "rank 0 failed to create shared-memory arena";
            return false;
        }

        MPI_Bcast(&name_len, 1, MPI_INT, 0, comm);
        if (name_len <= 0 || name_len >= 240)
        {
            last_error_ = "invalid shared-memory name length " + std::to_string(name_len);
            LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
            teardownSharedMemory();
            return false;
        }

        if (my_rank_ != 0)
        {
            shm_name_.assign(static_cast<size_t>(name_len), '\0');
        }
        MPI_Bcast(shm_name_.data(), name_len, MPI_CHAR, 0, comm);

        if (my_rank_ != 0)
        {
            shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0);
            if (shm_fd_ < 0)
            {
                last_error_ = "shm_open(open) failed for " + shm_name_ + ": " + std::string(strerror(errno));
                LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
            }
        }

        void *ptr = MAP_FAILED;
        if (shm_fd_ >= 0)
        {
            ptr = mmap(nullptr, arena_size_,
                       PROT_READ | PROT_WRITE, MAP_SHARED,
                       shm_fd_, 0);

            if (ptr == MAP_FAILED)
            {
                last_error_ = "mmap failed for " + shm_name_ + ": " + std::string(strerror(errno));
                LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
                close(shm_fd_);
                shm_fd_ = -1;
            }
            else
            {
                arena_ = static_cast<ShmemSpinArena *>(ptr);
            }
        }

        int local_ready = (arena_ != nullptr) ? 1 : 0;
        int all_ready = 0;
        MPI_Allreduce(&local_ready, &all_ready, 1, MPI_INT, MPI_MIN, comm);
        if (!all_ready)
        {
            if (last_error_.empty())
                last_error_ = "one or more ranks failed to map shared-memory arena " + shm_name_;
            LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
            teardownSharedMemory();
            return false;
        }

        // Rank 0 initializes only metadata. Touching the complete arena here
        // would place every persistent payload page on NUMA node zero and make
        // every other socket perform remote writes for the backend's lifetime.
        if (my_rank_ == 0)
        {
            const size_t metadata_bytes =
                sizeof(ShmemSpinArena) +
                static_cast<size_t>(num_ranks_) * sizeof(ShmemSpinArena::EpochSlot);
            std::memset(arena_, 0, metadata_bytes);
            arena_->num_ranks = num_ranks_;
        }

        // Publish num_ranks before buffer_at() computes participant offsets.
        if (MPI_Barrier(comm) != MPI_SUCCESS)
        {
            last_error_ = "MPI_Barrier failed after shared-memory initialization";
            LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
            teardownSharedMemory();
            return false;
        }

        // First-touch each rank's payload region from the socket that owns and
        // writes it during inference. Linux then backs those pages from the
        // appropriate NUMA node instead of concentrating the arena on rank 0.
        std::memset(
            arena_->buffer_at(my_rank_),
            0,
            ShmemSpinArena::CHUNK_CAPACITY * sizeof(float));

        if (MPI_Barrier(comm) != MPI_SUCCESS)
        {
            last_error_ = "MPI_Barrier failed after shared-memory NUMA first-touch";
            LOG_ERROR("ShmemSpinBackend::setupSharedMemory - " << last_error_);
            teardownSharedMemory();
            return false;
        }

        if (my_rank_ == 0)
        {
            if (shm_unlink(shm_name_.c_str()) == 0 || errno == ENOENT)
            {
                shm_unlinked_ = true;
            }
            else
            {
                LOG_WARN("ShmemSpinBackend::setupSharedMemory - shm_unlink(" << shm_name_
                                                                              << ") failed after mmap: " << strerror(errno));
            }
        }

        LOG_DEBUG("ShmemSpinBackend: Shared memory mapped at " << ptr
                                                               << " (" << arena_size_ << " bytes)"
                                                               << " for rank " << my_rank_
                                                               << " of " << num_ranks_
                                                               << " using " << shm_name_);
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

        // Rank 0 normally unlinks immediately after all ranks map the segment.
        // Retry here only for failures that occurred before that point.
        if (my_rank_ == 0 && !shm_name_.empty() && !shm_unlinked_)
        {
            if (shm_unlink(shm_name_.c_str()) == 0 || errno == ENOENT)
            {
                shm_unlinked_ = true;
            }
            else
            {
                LOG_WARN("ShmemSpinBackend::teardownSharedMemory - shm_unlink(" << shm_name_
                                                                                 << ") failed: " << strerror(errno));
            }
        }

        shm_name_.clear();
    }

    // =========================================================================
    // Fast-Path Check
    // =========================================================================

    bool ShmemSpinBackend::usesSharedMemoryAllreduce(
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op) const
    {
        (void)count;
        if (op != CollectiveOp::ALLREDUCE_SUM)
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

        // The native SUM path is total over payload size. Large tensors are
        // reduced in bounded chunks so geometry never selects a different
        // protocol merely because it crossed an arena-size threshold.
        if (usesSharedMemoryAllreduce(count, dtype, op))
        {
            // Element size depends on dtype; the support predicate admits only
            // these three representations.
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
                LLAMINAR_UNREACHABLE("shared-memory allreduce passed unsupported dtype");
            }

            if (count == 0 || num_ranks_ == 1)
                return true;
            if (!buffer)
            {
                last_error_ = "Shared-memory allreduce received a null non-empty payload";
                LOG_ERROR("ShmemSpinBackend::allreduce - " << last_error_);
                return false;
            }

            auto *payload = static_cast<std::byte *>(buffer);
            const auto publish_epoch = [&](const char *phase,
                                           size_t chunk_count) {
                ++my_epoch_;
                arena_->epoch_at(my_rank_)->epoch.store(
                    my_epoch_,
                    std::memory_order_release);
                for (int rank = 0; rank < num_ranks_; ++rank)
                {
                    if (rank == my_rank_)
                        continue;
                    if (!waitForPeerEpoch(
                            rank,
                            my_epoch_,
                            phase,
                            chunk_count))
                    {
                        return false;
                    }
                }
                return true;
            };

            const auto reduce_range = [&](
                                          void *chunk_output,
                                          size_t range_begin,
                                          size_t range_count) {
                switch (dtype)
                {
                case CollectiveDataType::FLOAT32:
                {
                    auto *out = static_cast<float *>(chunk_output) + range_begin;
                    reduce(
                        out,
                        arena_->buffer_at(0) + range_begin,
                        arena_->buffer_at(1) + range_begin,
                        range_count);
                    for (int rank = 2; rank < num_ranks_; ++rank)
                    {
                        reduce(
                            out,
                            out,
                            arena_->buffer_at(rank) + range_begin,
                            range_count);
                    }
                    return;
                }
                case CollectiveDataType::FLOAT16:
                {
                    auto *out = static_cast<uint16_t *>(chunk_output) + range_begin;
                    const auto *buffer0 =
                        reinterpret_cast<const uint16_t *>(arena_->buffer_at(0)) + range_begin;
                    const auto *buffer1 =
                        reinterpret_cast<const uint16_t *>(arena_->buffer_at(1)) + range_begin;
                    reduce_fp16(out, buffer0, buffer1, range_count);
                    for (int rank = 2; rank < num_ranks_; ++rank)
                    {
                        const auto *rank_buffer =
                            reinterpret_cast<const uint16_t *>(arena_->buffer_at(rank)) + range_begin;
                        reduce_fp16(out, out, rank_buffer, range_count);
                    }
                    return;
                }
                case CollectiveDataType::BFLOAT16:
                {
                    auto *out = static_cast<uint16_t *>(chunk_output) + range_begin;
                    const auto *buffer0 =
                        reinterpret_cast<const uint16_t *>(arena_->buffer_at(0)) + range_begin;
                    const auto *buffer1 =
                        reinterpret_cast<const uint16_t *>(arena_->buffer_at(1)) + range_begin;
                    reduce_bf16(out, buffer0, buffer1, range_count);
                    for (int rank = 2; rank < num_ranks_; ++rank)
                    {
                        const auto *rank_buffer =
                            reinterpret_cast<const uint16_t *>(arena_->buffer_at(rank)) + range_begin;
                        reduce_bf16(out, out, rank_buffer, range_count);
                    }
                    return;
                }
                default:
                    LLAMINAR_UNREACHABLE(
                        "shared-memory allreduce reduction reached unsupported dtype");
                }
            };

            constexpr size_t kParallelThresholdElements = 64u * 1024u;
            constexpr size_t kWorkSliceElements = 16u * 1024u;
            const bool parallelize =
                count >= kParallelThresholdElements && omp_in_parallel() == 0;

            if (parallelize)
            {
                bool protocol_ok = true;
#pragma omp parallel shared(payload, protocol_ok)
                {
                    for (size_t chunk_offset = 0;
                         chunk_offset < count && protocol_ok;)
                    {
                        const size_t chunk_count = std::min(
                            ShmemSpinArena::CHUNK_CAPACITY,
                            count - chunk_offset);
                        void *chunk_output = payload + chunk_offset * elem_size;
                        const size_t work_slices =
                            (chunk_count + kWorkSliceElements - 1u) /
                            kWorkSliceElements;

#pragma omp for schedule(static)
                        for (size_t slice = 0; slice < work_slices; ++slice)
                        {
                            const size_t range_begin = slice * kWorkSliceElements;
                            const size_t range_count = std::min(
                                kWorkSliceElements,
                                chunk_count - range_begin);
                            std::memcpy(
                                reinterpret_cast<std::byte *>(arena_->buffer_at(my_rank_)) +
                                    range_begin * elem_size,
                                static_cast<std::byte *>(chunk_output) +
                                    range_begin * elem_size,
                                range_count * elem_size);
                        }

#pragma omp master
                        {
                            protocol_ok = publish_epoch(
                                "entering parallel chunked allreduce",
                                chunk_count);
                        }
#pragma omp barrier

                        if (protocol_ok)
                        {
#pragma omp for schedule(static)
                            for (size_t slice = 0; slice < work_slices; ++slice)
                            {
                                const size_t range_begin = slice * kWorkSliceElements;
                                const size_t range_count = std::min(
                                    kWorkSliceElements,
                                    chunk_count - range_begin);
                                reduce_range(
                                    chunk_output,
                                    range_begin,
                                    range_count);
                            }
                        }

                        // No participant may overwrite its sole staging buffer
                        // until every peer has completed the reduction reads.
#pragma omp master
                        {
                            if (protocol_ok)
                            {
                                protocol_ok = publish_epoch(
                                    "leaving parallel chunked allreduce",
                                    chunk_count);
                            }
                        }
#pragma omp barrier

                        chunk_offset += chunk_count;
                    }
                }

                return protocol_ok;
            }

            for (size_t chunk_offset = 0; chunk_offset < count;)
            {
                const size_t chunk_count = std::min(
                    ShmemSpinArena::CHUNK_CAPACITY,
                    count - chunk_offset);
                void *chunk_output = payload + chunk_offset * elem_size;
                std::memcpy(
                    arena_->buffer_at(my_rank_),
                    chunk_output,
                    chunk_count * elem_size);
                if (!publish_epoch(
                        "entering chunked allreduce",
                        chunk_count))
                    return false;

                reduce_range(chunk_output, 0, chunk_count);
                if (!publish_epoch(
                        "leaving chunked allreduce",
                        chunk_count))
                    return false;
                chunk_offset += chunk_count;
            }
            return true;
        }

        // Other collective semantics are implemented by the general UPI backend.
        if (general_backend_)
        {
            return general_backend_->allreduce(buffer, count, dtype, op);
        }

        last_error_ = "No general backend for non-SUM allreduce";
        return false;
    }

    bool ShmemSpinBackend::allgather(const void *send_buf, void *recv_buf,
                                     size_t send_count, CollectiveDataType dtype)
    {
        if (general_backend_)
        {
            return general_backend_->allgather(send_buf, recv_buf, send_count, dtype);
        }
        last_error_ = "No general backend for allgather";
        return false;
    }

    bool ShmemSpinBackend::allgatherv(const void *send_buf, size_t send_count,
                                      void *recv_buf,
                                      const std::vector<int> &recv_counts,
                                      const std::vector<int> &displacements,
                                      CollectiveDataType dtype)
    {
        if (general_backend_)
        {
            return general_backend_->allgatherv(send_buf, send_count, recv_buf,
                                                recv_counts, displacements, dtype);
        }
        last_error_ = "No general backend for allgatherv";
        return false;
    }

    bool ShmemSpinBackend::reduceScatter(const void *send_buf, void *recv_buf,
                                         size_t recv_count, CollectiveDataType dtype,
                                         CollectiveOp op)
    {
        if (general_backend_)
        {
            return general_backend_->reduceScatter(send_buf, recv_buf, recv_count, dtype, op);
        }
        last_error_ = "No general backend for reduceScatter";
        return false;
    }

    bool ShmemSpinBackend::gatherVariableFloatRecordsToRoot(
        const float *local_records,
        size_t local_record_count,
        float *root_records,
        size_t root_record_capacity,
        size_t record_width_elements,
        int root_rank,
        size_t &gathered_record_count)
    {
        gathered_record_count = 0u;
        if (abort_requested_.load(std::memory_order_acquire))
        {
            last_error_ = "ShmemSpinBackend abort has been requested";
            return false;
        }
        if (!initialized_ || !arena_ || num_ranks_ <= 1 || root_rank < 0 ||
            root_rank >= num_ranks_ || my_rank_ < 0 ||
            my_rank_ >= num_ranks_ || record_width_elements == 0u ||
            root_record_capacity == 0u ||
            local_record_count > root_record_capacity ||
            (local_record_count > 0u && !local_records) ||
            (my_rank_ == root_rank && !root_records) ||
            rooted_publication_element_counts_.size() !=
                static_cast<size_t>(num_ranks_) ||
            rooted_publication_element_offsets_.size() !=
                static_cast<size_t>(num_ranks_) ||
            root_record_capacity >
                std::numeric_limits<size_t>::max() / record_width_elements ||
            local_record_count >
                std::numeric_limits<size_t>::max() / record_width_elements)
        {
            last_error_ =
                "invalid shared-memory rooted packed-record contract";
            LOG_ERROR("ShmemSpinBackend::gatherVariableFloatRecordsToRoot - "
                      << last_error_ << " domain=" << domain_id_
                      << " rank=" << my_rank_ << "/" << num_ranks_
                      << " root=" << root_rank
                      << " local_records=" << local_record_count
                      << " capacity=" << root_record_capacity
                      << " record_width=" << record_width_elements);
            return false;
        }

        const size_t local_elements =
            local_record_count * record_width_elements;
        auto *const my_slot = arena_->epoch_at(my_rank_);

        /*
         * The size publication is a transaction header. Every participant
         * waits for every header before computing the common chunk count, so
         * ranks with no local routes still execute exactly the same epoch DAG.
         */
        my_slot->payload_elements.store(
            static_cast<uint64_t>(local_elements),
            std::memory_order_relaxed);
        ++my_epoch_;
        my_slot->epoch.store(my_epoch_, std::memory_order_release);
        for (int rank = 0; rank < num_ranks_; ++rank)
        {
            if (rank == my_rank_)
                continue;
            if (!waitForPeerEpoch(
                    rank,
                    my_epoch_,
                    "publishing rooted packed-record sizes",
                    local_elements))
            {
                return false;
            }
        }

        size_t total_elements = 0u;
        size_t maximum_participant_elements = 0u;
        for (int rank = 0; rank < num_ranks_; ++rank)
        {
            const uint64_t rank_elements_u64 =
                arena_->epoch_at(rank)->payload_elements.load(
                    std::memory_order_relaxed);
            if (rank_elements_u64 >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
            {
                last_error_ =
                    "rooted packed-record element count exceeds size_t";
                LOG_ERROR("ShmemSpinBackend::gatherVariableFloatRecordsToRoot - "
                          << last_error_ << " source=" << rank);
                abort();
                return false;
            }
            const size_t rank_elements =
                static_cast<size_t>(rank_elements_u64);
            if (rank_elements % record_width_elements != 0u ||
                rank_elements >
                    root_record_capacity * record_width_elements -
                        total_elements)
            {
                last_error_ =
                    "rooted packed-record sizes are malformed or exceed capacity";
                LOG_ERROR("ShmemSpinBackend::gatherVariableFloatRecordsToRoot - "
                          << last_error_ << " source=" << rank
                          << " source_elements=" << rank_elements
                          << " accumulated_elements=" << total_elements
                          << " capacity_elements="
                          << root_record_capacity * record_width_elements);
                abort();
                return false;
            }
            rooted_publication_element_counts_[static_cast<size_t>(rank)] =
                rank_elements;
            total_elements += rank_elements;
            maximum_participant_elements =
                std::max(maximum_participant_elements, rank_elements);
        }

        /*
         * Close the transaction-header lifetime before any participant may
         * return or publish the next header. The first epoch above proves that
         * every count is available; this second epoch proves that every rank
         * has consumed every count. Both edges are required because the count
         * field is a single persistent slot rather than an epoch-indexed ring.
         *
         * This is especially important for an all-empty transaction. Such a
         * transaction has no payload chunk whose ready/consumed handshake could
         * otherwise delay slot reuse. Without this explicit consumed edge, a
         * fast participant can publish transaction N+1 while a slower root is
         * still reading transaction N and the root can observe a stale/future
         * count under an epoch that it has already acquired.
         */
        ++my_epoch_;
        my_slot->epoch.store(my_epoch_, std::memory_order_release);
        for (int rank = 0; rank < num_ranks_; ++rank)
        {
            if (rank == my_rank_)
                continue;
            if (!waitForPeerEpoch(
                    rank,
                    my_epoch_,
                    "retiring rooted packed-record size headers",
                    total_elements))
            {
                return false;
            }
        }

        /* Root's contribution is always the first packed block. */
        size_t next_offset =
            rooted_publication_element_counts_[
                static_cast<size_t>(root_rank)];
        rooted_publication_element_offsets_[static_cast<size_t>(root_rank)] = 0u;
        for (int rank = 0; rank < num_ranks_; ++rank)
        {
            if (rank == root_rank)
                continue;
            rooted_publication_element_offsets_[static_cast<size_t>(rank)] =
                next_offset;
            next_offset +=
                rooted_publication_element_counts_[static_cast<size_t>(rank)];
        }
        if (next_offset != total_elements)
        {
            last_error_ = "rooted packed-record offset construction diverged";
            LOG_ERROR("ShmemSpinBackend::gatherVariableFloatRecordsToRoot - "
                      << last_error_ << " offsets=" << next_offset
                      << " elements=" << total_elements);
            abort();
            return false;
        }

        if (my_rank_ == root_rank && local_elements > 0u &&
            local_records != root_records)
        {
            std::memmove(
                root_records,
                local_records,
                local_elements * sizeof(float));
        }

        for (size_t chunk_offset = 0u;
             chunk_offset < maximum_participant_elements;
             chunk_offset += ShmemSpinArena::CHUNK_CAPACITY)
        {
            const size_t my_chunk_elements =
                chunk_offset < local_elements
                    ? std::min(
                          ShmemSpinArena::CHUNK_CAPACITY,
                          local_elements - chunk_offset)
                    : 0u;
            if (my_rank_ != root_rank && my_chunk_elements > 0u)
            {
                std::memcpy(
                    arena_->buffer_at(my_rank_),
                    local_records + chunk_offset,
                    my_chunk_elements * sizeof(float));
            }

            ++my_epoch_;
            my_slot->epoch.store(my_epoch_, std::memory_order_release);
            if (my_rank_ == root_rank)
            {
                for (int rank = 0; rank < num_ranks_; ++rank)
                {
                    if (rank == root_rank)
                        continue;
                    if (!waitForPeerEpoch(
                            rank,
                            my_epoch_,
                            "waiting for a packed-record chunk",
                            total_elements))
                    {
                        return false;
                    }
                }

                for (int rank = 0; rank < num_ranks_; ++rank)
                {
                    if (rank == root_rank)
                        continue;
                    const size_t rank_elements =
                        rooted_publication_element_counts_[
                            static_cast<size_t>(rank)];
                    const size_t rank_chunk_elements =
                        chunk_offset < rank_elements
                            ? std::min(
                                  ShmemSpinArena::CHUNK_CAPACITY,
                                  rank_elements - chunk_offset)
                            : 0u;
                    if (rank_chunk_elements == 0u)
                        continue;
                    std::memcpy(
                        root_records +
                            rooted_publication_element_offsets_[
                                static_cast<size_t>(rank)] +
                            chunk_offset,
                        arena_->buffer_at(rank),
                        rank_chunk_elements * sizeof(float));
                }

                // Release every peer to reuse its sole staging buffer.
                ++my_epoch_;
                my_slot->epoch.store(my_epoch_, std::memory_order_release);
            }
            else
            {
                const uint64_t consumed_epoch = my_epoch_ + 1u;
                if (!waitForPeerEpoch(
                        root_rank,
                        consumed_epoch,
                        "waiting for root to consume a packed-record chunk",
                        my_chunk_elements))
                {
                    return false;
                }
                ++my_epoch_;
                my_slot->epoch.store(my_epoch_, std::memory_order_release);
            }
        }

        /*
         * Root waits for the final peer acknowledgement so successful return
         * means the complete transaction, not merely root's local copy, is done.
         */
        if (my_rank_ == root_rank && maximum_participant_elements > 0u)
        {
            for (int rank = 0; rank < num_ranks_; ++rank)
            {
                if (rank == root_rank)
                    continue;
                if (!waitForPeerEpoch(
                        rank,
                        my_epoch_,
                        "completing rooted packed-record publication",
                        total_elements))
                {
                    return false;
                }
            }
        }

        if (my_rank_ == root_rank)
            gathered_record_count = total_elements / record_width_elements;
        return true;
    }

    bool ShmemSpinBackend::broadcast(void *buffer, size_t count,
                                     CollectiveDataType dtype, int root_rank)
    {
        if (abort_requested_.load(std::memory_order_acquire))
        {
            last_error_ = "ShmemSpinBackend abort has been requested";
            return false;
        }
        const size_t element_size = collectiveElementSize(dtype);
        if (!initialized_ || !arena_ || root_rank < 0 ||
            root_rank >= num_ranks_ || my_rank_ < 0 ||
            my_rank_ >= num_ranks_ || element_size == 0u ||
            (count > 0u && !buffer) ||
            count > std::numeric_limits<size_t>::max() / element_size)
        {
            last_error_ = "invalid native shared-memory broadcast contract";
            LOG_ERROR("ShmemSpinBackend::broadcast - " << last_error_
                      << " domain=" << domain_id_
                      << " rank=" << my_rank_ << "/" << num_ranks_
                      << " root=" << root_rank << " count=" << count
                      << " element_size=" << element_size);
            return false;
        }
        if (count == 0u || num_ranks_ == 1)
            return true;

        constexpr size_t kArenaBytes =
            ShmemSpinArena::CHUNK_CAPACITY * sizeof(float);
        const size_t chunk_capacity_elements = kArenaBytes / element_size;
        auto *const payload = static_cast<std::byte *>(buffer);
        auto *const root_staging = reinterpret_cast<std::byte *>(
            arena_->buffer_at(root_rank));
        auto *const my_slot = arena_->epoch_at(my_rank_);

        for (size_t chunk_offset = 0u; chunk_offset < count;)
        {
            const size_t chunk_elements = std::min(
                chunk_capacity_elements,
                count - chunk_offset);
            const size_t chunk_bytes = chunk_elements * element_size;
            const size_t byte_offset = chunk_offset * element_size;
            const uint64_t transaction_epoch = my_epoch_ + 1u;

            if (my_rank_ == root_rank)
            {
                std::memcpy(
                    root_staging,
                    payload + byte_offset,
                    chunk_bytes);
                ++my_epoch_;
                my_slot->epoch.store(my_epoch_, std::memory_order_release);
                for (int rank = 0; rank < num_ranks_; ++rank)
                {
                    if (rank == root_rank)
                        continue;
                    if (!waitForPeerEpoch(
                            rank,
                            transaction_epoch,
                            "waiting for shared-memory broadcast consumption",
                            chunk_elements))
                    {
                        return false;
                    }
                }
            }
            else
            {
                if (!waitForPeerEpoch(
                        root_rank,
                        transaction_epoch,
                        "waiting for shared-memory broadcast publication",
                        chunk_elements))
                {
                    return false;
                }
                std::memcpy(
                    payload + byte_offset,
                    root_staging,
                    chunk_bytes);
                ++my_epoch_;
                my_slot->epoch.store(my_epoch_, std::memory_order_release);
            }
            chunk_offset += chunk_elements;
        }
        return true;
    }

    bool ShmemSpinBackend::synchronize()
    {
        if (general_backend_)
        {
            return general_backend_->synchronize();
        }
        // Without a general backend, use the native epoch-based N-way barrier.
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
