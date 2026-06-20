/**
 * @file UPIBackend.cpp
 * @brief UPI-based collective backend implementation
 *
 * Implements cross-socket CPU tensor parallelism using MPI over UPI.
 * Uses domain-specific MPI communicators for isolated collective operations.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "UPIBackend.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include "../../utils/NodeTopology.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

namespace llaminar2
{
    namespace
    {
        bool waitForCollectiveRequest(
            MPI_Request &request,
            int timeout_ms,
            const std::string &operation,
            int domain_rank,
            int domain_size,
            size_t count,
            std::string &last_error)
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
            int complete = 0;
            int result = MPI_SUCCESS;
            while (!complete)
            {
                result = MPI_Test(&request, &complete, MPI_STATUS_IGNORE);
                if (result != MPI_SUCCESS || complete)
                    break;

                if (std::chrono::steady_clock::now() >= deadline)
                {
                    last_error = operation + " timed out after " + std::to_string(timeout_ms) +
                                 "ms on domain rank " + std::to_string(domain_rank) +
                                 "/" + std::to_string(domain_size) +
                                 " (count=" + std::to_string(count) + ")";
                    LOG_ERROR("UPICollectiveBackend - " << last_error
                              << "; aborting MPI job to avoid rank desynchronization");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return false;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            if (result != MPI_SUCCESS)
            {
                last_error = operation + " failed with code " + std::to_string(result);
                LOG_ERROR("UPICollectiveBackend - " << last_error);
                return false;
            }

            return true;
        }
    } // namespace

    // =============================================================================
    // Constructor / Destructor
    // =============================================================================

    UPICollectiveBackend::UPICollectiveBackend(MPI_Comm domain_comm,
                                               const NodeTopology *topology)
        : domain_comm_(domain_comm), domain_rank_(-1), domain_size_(0), bandwidth_gbps_(estimateBandwidth(topology))
    {
        if (domain_comm_ != MPI_COMM_NULL)
        {
            MPI_Comm_rank(domain_comm_, &domain_rank_);
            MPI_Comm_size(domain_comm_, &domain_size_);

            LOG_DEBUG("UPICollectiveBackend: Created with domain_rank=" << domain_rank_
                                                                        << ", domain_size=" << domain_size_
                                                                        << ", estimated_bandwidth=" << bandwidth_gbps_ << " GB/s");
        }
        else
        {
            LOG_WARN("UPICollectiveBackend: Created with MPI_COMM_NULL communicator");
        }
    }

    UPICollectiveBackend::~UPICollectiveBackend()
    {
        shutdown();
        // NOTE: We do NOT free domain_comm_ here - the caller owns it
        // (typically TPDomainBuilder or similar creates and manages the communicator)
    }

    // =============================================================================
    // Capability Queries
    // =============================================================================

    bool UPICollectiveBackend::supportsDevice(DeviceType type) const
    {
        // UPI operates on host memory only - CPU is directly supported
        return type == DeviceType::CPU;
    }

    bool UPICollectiveBackend::supportsDirectTransfer(DeviceId src, DeviceId dst) const
    {
        // UPI can only directly transfer between CPU buffers
        // Both should ideally be NUMA-local for optimal performance
        return src.type == DeviceType::CPU && dst.type == DeviceType::CPU;
    }

    bool UPICollectiveBackend::isAvailable() const
    {
        // Available if we have a valid communicator with at least 1 rank
        return domain_comm_ != MPI_COMM_NULL && domain_size_ > 0;
    }

    // =============================================================================
    // Lifecycle
    // =============================================================================

    bool UPICollectiveBackend::initialize(const DeviceGroup &group)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "Cannot initialize UPI backend with null communicator";
            LOG_ERROR("UPICollectiveBackend::initialize - " << last_error_);
            return false;
        }

        // Validate group scope - UPI is used for LOCAL scope (same node, cross-socket)
        if (group.scope != CollectiveScope::LOCAL)
        {
            LOG_WARN("UPICollectiveBackend: Expected LOCAL scope for cross-socket CPU TP, got "
                     << (group.scope == CollectiveScope::GLOBAL ? "GLOBAL" : "unknown"));
        }

        group_ = group;
        initialized_ = true;

        LOG_DEBUG("UPICollectiveBackend initialized for group '" << group.name
                                                                 << "' with " << domain_size_ << " ranks over UPI");

        return true;
    }

    bool UPICollectiveBackend::isInitialized() const
    {
        return initialized_;
    }

    void UPICollectiveBackend::shutdown()
    {
        if (initialized_)
        {
            LOG_DEBUG("UPICollectiveBackend shutdown");
        }
        initialized_ = false;
        // NOTE: We do NOT free domain_comm_ - caller owns it
    }

    // =============================================================================
    // Collective Operations
    // =============================================================================

    bool UPICollectiveBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        // FP16/BF16 reductions require special handling — MPI has no native
        // half-precision reduction ops. Use allgather + local FP32 reduce.
        if ((dtype == CollectiveDataType::FLOAT16 || dtype == CollectiveDataType::BFLOAT16) &&
            (op == CollectiveOp::ALLREDUCE_SUM || op == CollectiveOp::ALLREDUCE_MAX || op == CollectiveOp::ALLREDUCE_MIN))
        {
            return allreduceHalfPrecision(buffer, count, dtype, op);
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);
        MPI_Op mpi_op = toMPIOp(op);

        int result = MPI_SUCCESS;
        const int timeout_ms = debugEnv().tp_collect_timeout_ms;
        if (timeout_ms > 0)
        {
            MPI_Request request = MPI_REQUEST_NULL;
            result = MPI_Iallreduce(
                MPI_IN_PLACE,
                buffer,
                static_cast<int>(count),
                mpi_dtype,
                mpi_op,
                domain_comm_,
                &request);

            if (result == MPI_SUCCESS)
            {
                const auto start = std::chrono::steady_clock::now();
                const auto deadline = start + std::chrono::milliseconds(timeout_ms);
                int complete = 0;
                while (!complete)
                {
                    result = MPI_Test(&request, &complete, MPI_STATUS_IGNORE);
                    if (result != MPI_SUCCESS || complete)
                        break;

                    if (std::chrono::steady_clock::now() >= deadline)
                    {
                        last_error_ = "MPI_Iallreduce timed out after " + std::to_string(timeout_ms) +
                                      "ms on domain rank " + std::to_string(domain_rank_) +
                                      "/" + std::to_string(domain_size_) +
                                      " (count=" + std::to_string(count) + ")";
                        LOG_ERROR("UPICollectiveBackend::allreduce - " << last_error_
                                  << "; aborting MPI job to avoid rank desynchronization");

                        // A timed-out nonblocking collective leaves the communicator in
                        // an unsafe state; continuing into later MPI calls can deadlock or
                        // corrupt ordering. Fail the MPI job deliberately and loudly.
                        MPI_Abort(MPI_COMM_WORLD, 1);
                        return false;
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }
        else
        {
            // Use MPI_IN_PLACE for efficient in-place allreduce.
            result = MPI_Allreduce(
                MPI_IN_PLACE,
                buffer,
                static_cast<int>(count),
                mpi_dtype,
                mpi_op,
                domain_comm_);
        }

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allreduce failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::allreduce - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);

        int result = MPI_Allgather(
            send_buf,
            static_cast<int>(send_count),
            mpi_dtype,
            recv_buf,
            static_cast<int>(send_count),
            mpi_dtype,
            domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allgather failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::allgather - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::allgatherv(
        const void *send_buf,
        size_t send_count,
        void *recv_buf,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        CollectiveDataType dtype)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);

        int result = MPI_Allgatherv(
            send_buf,
            static_cast<int>(send_count),
            mpi_dtype,
            recv_buf,
            recv_counts.data(),
            displacements.data(),
            mpi_dtype,
            domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allgatherv failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::allgatherv - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);
        MPI_Op mpi_op = toMPIOp(op);

        // MPI_Reduce_scatter requires an array of recvcounts (one per rank)
        // For uniform distribution, all ranks get the same count
        std::vector<int> recvcounts(domain_size_, static_cast<int>(recv_count));

        int result = MPI_Reduce_scatter(
            send_buf,
            recv_buf,
            recvcounts.data(),
            mpi_dtype,
            mpi_op,
            domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Reduce_scatter failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::reduceScatter - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root_rank)
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        MPI_Datatype mpi_dtype = toMPIDatatype(dtype);

        int result = MPI_SUCCESS;
        const int timeout_ms = debugEnv().tp_collect_timeout_ms;
        if (timeout_ms > 0)
        {
            MPI_Request request = MPI_REQUEST_NULL;
            result = MPI_Ibcast(
                buffer,
                static_cast<int>(count),
                mpi_dtype,
                root_rank,
                domain_comm_,
                &request);
            if (result == MPI_SUCCESS &&
                !waitForCollectiveRequest(request,
                                          timeout_ms,
                                          "MPI_Ibcast",
                                          domain_rank_,
                                          domain_size_,
                                          count,
                                          last_error_))
            {
                return false;
            }
        }
        else
        {
            result = MPI_Bcast(
                buffer,
                static_cast<int>(count),
                mpi_dtype,
                root_rank,
                domain_comm_);
        }

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Bcast failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::broadcast - " << last_error_);
            return false;
        }

        return true;
    }

    bool UPICollectiveBackend::synchronize()
    {
        if (domain_comm_ == MPI_COMM_NULL)
        {
            last_error_ = "UPI backend has null communicator";
            return false;
        }

        if (!initialized_)
        {
            last_error_ = "UPI backend not initialized";
            return false;
        }

        int result = MPI_SUCCESS;
        const int timeout_ms = debugEnv().tp_collect_timeout_ms;
        if (timeout_ms > 0)
        {
            MPI_Request request = MPI_REQUEST_NULL;
            result = MPI_Ibarrier(domain_comm_, &request);
            if (result == MPI_SUCCESS &&
                !waitForCollectiveRequest(request,
                                          timeout_ms,
                                          "MPI_Ibarrier",
                                          domain_rank_,
                                          domain_size_,
                                          0,
                                          last_error_))
            {
                return false;
            }
        }
        else
        {
            result = MPI_Barrier(domain_comm_);
        }

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Barrier failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::synchronize - " << last_error_);
            return false;
        }

        return true;
    }

    // =============================================================================
    // Static Helpers
    // =============================================================================

    MPI_Datatype UPICollectiveBackend::toMPIDatatype(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return MPI_FLOAT;
        case CollectiveDataType::FLOAT16:
            // FP16 doesn't have native MPI type - use MPI_UINT16_T for raw bytes
            // Note: Reduction operations (SUM/MAX/MIN) won't work correctly for FP16
            // with this approach - would need custom MPI_Op for true FP16 reduction
            return MPI_UINT16_T;
        case CollectiveDataType::BFLOAT16:
            // BF16 doesn't have native MPI type - use MPI_UINT16_T for raw bytes
            // Same caveat as FP16 for reduction operations
            return MPI_UINT16_T;
        case CollectiveDataType::INT32:
            return MPI_INT;
        case CollectiveDataType::INT8:
            return MPI_INT8_T;
        default:
            LOG_WARN("UPICollectiveBackend::toMPIDatatype - Unknown dtype, defaulting to MPI_BYTE");
            return MPI_BYTE;
        }
    }

    MPI_Op UPICollectiveBackend::toMPIOp(CollectiveOp op)
    {
        switch (op)
        {
        case CollectiveOp::ALLREDUCE_SUM:
            return MPI_SUM;
        case CollectiveOp::ALLREDUCE_MAX:
            return MPI_MAX;
        case CollectiveOp::ALLREDUCE_MIN:
            return MPI_MIN;
        default:
            // For non-reduction ops (ALLGATHER, BROADCAST), return SUM as default
            // (caller should not use reduction op for non-reduction collectives)
            return MPI_SUM;
        }
    }

    // =========================================================================
    // Half-Precision Allreduce (allgather + local FP32 reduce)
    // =========================================================================

    namespace
    {
        // BF16 ↔ FP32: upper 16 bits of IEEE-754 float
        inline float upi_bf16_to_float(uint16_t bf)
        {
            uint32_t f = static_cast<uint32_t>(bf) << 16;
            float result;
            std::memcpy(&result, &f, sizeof(float));
            return result;
        }

        inline uint16_t upi_float_to_bf16(float val)
        {
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(float));
            return static_cast<uint16_t>(bits >> 16); // truncation
        }

        // FP16 ↔ FP32: software IEEE-754 half-precision
        inline float upi_fp16_to_float(uint16_t h)
        {
            uint32_t sign = static_cast<uint32_t>(h >> 15) << 31;
            uint32_t exp = (h >> 10) & 0x1Fu;
            uint32_t mant = h & 0x3FFu;
            uint32_t f;

            if (exp == 0)
            {
                if (mant == 0)
                    f = sign;
                else
                {
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
                f = sign | 0x7F800000u | (mant << 13);
            else
                f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);

            float result;
            std::memcpy(&result, &f, sizeof(float));
            return result;
        }

        inline uint16_t upi_float_to_fp16(float val)
        {
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(float));
            uint32_t sign = (bits >> 16) & 0x8000u;
            int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
            uint32_t mant = bits & 0x7FFFFFu;

            if (exp <= 0)
            {
                if (exp < -10)
                    return static_cast<uint16_t>(sign);
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
                    return static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13));
                return static_cast<uint16_t>(sign | 0x7C00u);
            }

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

    bool UPICollectiveBackend::allreduceHalfPrecision(
        void *buffer, size_t count, CollectiveDataType dtype, CollectiveOp op)
    {
        // Step 1: Allgather raw half-precision data from all ranks
        // (MPI_Allgather with MPI_UINT16_T is safe — just raw byte transport)
        std::vector<uint16_t> gathered(count * static_cast<size_t>(domain_size_));

        int result = MPI_Allgather(
            buffer,
            static_cast<int>(count),
            MPI_UINT16_T,
            gathered.data(),
            static_cast<int>(count),
            MPI_UINT16_T,
            domain_comm_);

        if (result != MPI_SUCCESS)
        {
            last_error_ = "MPI_Allgather (half-precision allreduce) failed with code " + std::to_string(result);
            LOG_ERROR("UPICollectiveBackend::allreduceHalfPrecision - " << last_error_);
            return false;
        }

        // Step 2: Local reduce in FP32 for precision
        auto *out = static_cast<uint16_t *>(buffer);
        const bool is_bf16 = (dtype == CollectiveDataType::BFLOAT16);

        // Convert first rank's data to FP32 accumulator
        std::vector<float> acc(count);
        const uint16_t *rank0 = gathered.data();
        for (size_t i = 0; i < count; ++i)
        {
            acc[i] = is_bf16 ? upi_bf16_to_float(rank0[i]) : upi_fp16_to_float(rank0[i]);
        }

        // Accumulate remaining ranks in FP32
        for (int r = 1; r < domain_size_; ++r)
        {
            const uint16_t *rank_data = gathered.data() + static_cast<size_t>(r) * count;
            if (op == CollectiveOp::ALLREDUCE_SUM)
            {
                for (size_t i = 0; i < count; ++i)
                    acc[i] += is_bf16 ? upi_bf16_to_float(rank_data[i]) : upi_fp16_to_float(rank_data[i]);
            }
            else if (op == CollectiveOp::ALLREDUCE_MAX)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    float v = is_bf16 ? upi_bf16_to_float(rank_data[i]) : upi_fp16_to_float(rank_data[i]);
                    if (v > acc[i])
                        acc[i] = v;
                }
            }
            else if (op == CollectiveOp::ALLREDUCE_MIN)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    float v = is_bf16 ? upi_bf16_to_float(rank_data[i]) : upi_fp16_to_float(rank_data[i]);
                    if (v < acc[i])
                        acc[i] = v;
                }
            }
        }

        // Step 3: Convert FP32 accumulator back to half-precision
        for (size_t i = 0; i < count; ++i)
        {
            out[i] = is_bf16 ? upi_float_to_bf16(acc[i]) : upi_float_to_fp16(acc[i]);
        }

        return true;
    }

    float UPICollectiveBackend::estimateBandwidth(const NodeTopology *topology)
    {
        // If we have topology info with inter-socket links, use that
        if (topology != nullptr && topology->numSockets() > 1)
        {
            // Get link info from topology if available
            // Default to Intel UPI bandwidth for now
            // AMD Infinity Fabric is typically faster (~100 GB/s)

            // Try to detect CPU vendor from /proc/cpuinfo
            std::ifstream cpuinfo("/proc/cpuinfo");
            if (cpuinfo.is_open())
            {
                std::string line;
                while (std::getline(cpuinfo, line))
                {
                    if (line.find("vendor_id") != std::string::npos)
                    {
                        if (line.find("AuthenticAMD") != std::string::npos)
                        {
                            LOG_DEBUG("UPICollectiveBackend: Detected AMD CPU, "
                                      "estimating Infinity Fabric bandwidth ~100 GB/s");
                            return 100.0f; // AMD Infinity Fabric
                        }
                        else if (line.find("GenuineIntel") != std::string::npos)
                        {
                            LOG_DEBUG("UPICollectiveBackend: Detected Intel CPU, "
                                      "estimating UPI bandwidth ~50 GB/s");
                            return 50.0f; // Intel UPI
                        }
                        break;
                    }
                }
            }
        }

        // Default to conservative Intel UPI estimate
        return 50.0f;
    }

} // namespace llaminar2
