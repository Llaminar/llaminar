/**
 * @file HeterogeneousBackend.cpp
 * @brief Implementation of HeterogeneousBackend for mixed CUDA+ROCm collectives
 *
 * This file implements device grouping logic and sub-backend management.
 * The actual collective operations (allreduce, etc.) are stubs for Phase 1.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "HeterogeneousBackend.h"
#include "NCCLBackend.h"
#include "RCCLBackend.h"
#include "PCIeBARBackend.h"
#include "../../utils/Logger.h"
#include "../../backends/rocm/ROCmBackend.h" // For ROCmBackend::deviceToDevice
#include "../../backends/BackendManager.h"   // For getBackend()

#include <algorithm>
#include <thread>

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>
#endif

// Note: We do NOT include hip/hip_runtime.h here because it conflicts with cuda_runtime.h.
// Instead, we use ROCmBackend's deviceToDevice() which wraps hipMemcpy internally.

namespace llaminar2
{

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    // ═══════════════════════════════════════════════════════════════════════════
    // Construction / Destruction
    // ═══════════════════════════════════════════════════════════════════════════

    HeterogeneousBackend::HeterogeneousBackend()
        : initialized_(false),
          cuda_bridge_(DeviceId::cpu()),
          rocm_bridge_(DeviceId::cpu())
    {
        LOG_DEBUG("HeterogeneousBackend: Created");
    }

    HeterogeneousBackend::~HeterogeneousBackend()
    {
        if (initialized_)
        {
            shutdown();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Capability Queries
    // ═══════════════════════════════════════════════════════════════════════════

    bool HeterogeneousBackend::supportsDirectTransfer(DeviceId src, DeviceId dst) const
    {
        // Direct transfer within same vendor domain
        if (src.type == dst.type)
        {
            return true;
        }

        // Cross-vendor via PCIeBAR bridge (only between bridge devices)
        if (initialized_ && bridge_backend_)
        {
            return bridge_backend_->supportsDirectTransfer(src, dst);
        }

        return false;
    }

    bool HeterogeneousBackend::isAvailable() const
    {
        // Check if PCIeBAR capability is available (required for cross-vendor)
        PCIeBARBackend probe;
        return probe.isAvailable();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ═══════════════════════════════════════════════════════════════════════════

    bool HeterogeneousBackend::initialize(const DeviceGroup &group)
    {
        if (initialized_)
        {
            LOG_WARN("HeterogeneousBackend: Already initialized");
            return true;
        }

        LOG_INFO("HeterogeneousBackend: Initializing with " << group.size() << " devices");

        // Validate the group
        if (!validateGroup(group))
        {
            return false;
        }

        // Store the device group
        device_group_ = group;

        // Partition devices by vendor
        if (!partitionDevices(group))
        {
            return false;
        }

        // Select bridge devices
        selectBridgeDevices();

        LOG_INFO("HeterogeneousBackend: CUDA devices: " << cuda_devices_.size()
                                                        << ", ROCm devices: " << rocm_devices_.size());
        LOG_INFO("HeterogeneousBackend: Bridge devices - CUDA: " << cuda_bridge_.toString()
                                                                 << ", ROCm: " << rocm_bridge_.toString());

        // Create sub-backends
        if (!createNCCLBackend())
        {
            return false;
        }

        if (!createRCCLBackend())
        {
            return false;
        }

        if (!createBridgeBackend())
        {
            return false;
        }

        initialized_ = true;
        LOG_INFO("HeterogeneousBackend: Initialization complete");
        return true;
    }

    void HeterogeneousBackend::shutdown()
    {
        if (!initialized_)
        {
            return;
        }

        LOG_DEBUG("HeterogeneousBackend: Shutting down");

        // Shutdown sub-backends in reverse order
        if (multi_pair_bridge_backend_)
        {
            multi_pair_bridge_backend_->shutdown();
            multi_pair_bridge_backend_.reset();
        }

        if (bridge_backend_)
        {
            bridge_backend_->shutdown();
            bridge_backend_.reset();
        }

        if (rccl_backend_)
        {
            rccl_backend_->shutdown();
            rccl_backend_.reset();
        }

        if (nccl_backend_)
        {
            nccl_backend_->shutdown();
            nccl_backend_.reset();
        }

        // Clear device lists and cached data
        cuda_devices_.clear();
        rocm_devices_.clear();
        bridge_pairs_.clear();
        cuda_bridge_ = DeviceId::cpu();
        rocm_bridge_ = DeviceId::cpu();

        initialized_ = false;
        LOG_DEBUG("HeterogeneousBackend: Shutdown complete");
    }

    bool HeterogeneousBackend::reserveTempBufferBytes(size_t bytes)
    {
        bool success = true;

        // Reserve in all sub-backends
        if (nccl_backend_)
        {
            success &= nccl_backend_->reserveTempBufferBytes(bytes);
        }

        if (rccl_backend_)
        {
            success &= rccl_backend_->reserveTempBufferBytes(bytes);
        }

        if (bridge_backend_)
        {
            success &= bridge_backend_->reserveTempBufferBytes(bytes);
        }

        return success;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Collective Operations (Stubs)
    // ═══════════════════════════════════════════════════════════════════════════

    bool HeterogeneousBackend::allreduce(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)op;

        // Single-buffer allreduce is not supported for heterogeneous configs
        // because we need separate buffers per device. Use allreduceMulti instead.
        last_error_ = "HeterogeneousBackend::allreduce single-buffer not supported, use allreduceMulti";
        LOG_ERROR(last_error_);
        return false;
    }

    // Helper function to compute element size for collective data types
    static size_t collectiveDataTypeSize(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return 4;
        case CollectiveDataType::FLOAT16:
        case CollectiveDataType::BFLOAT16:
            return 2;
        case CollectiveDataType::INT32:
            return 4;
        case CollectiveDataType::INT8:
            return 1;
        default:
            return 4; // Default to float32 size
        }
    }

    bool HeterogeneousBackend::allreduceMulti(
        const std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        // Validate buffer count matches total device count
        size_t expected_buffers = device_group_.devices.size();
        if (buffers.size() != expected_buffers)
        {
            last_error_ = "Buffer count (" + std::to_string(buffers.size()) +
                          ") does not match device count (" +
                          std::to_string(expected_buffers) + ")";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        // Split buffers into CUDA and ROCm groups based on device order
        // The buffers are in device_group_.devices order, so we need to map
        // buffer indices to CUDA/ROCm based on each device's type
        std::vector<void *> cuda_buffers;
        std::vector<void *> rocm_buffers;
        void *cuda_bridge_buf = nullptr;
        void *rocm_bridge_buf = nullptr;

        for (size_t i = 0; i < device_group_.devices.size(); ++i)
        {
            const DeviceId &device = device_group_.devices[i];
            void *buf = buffers[i];

            if (device.is_cuda())
            {
                cuda_buffers.push_back(buf);
                // Bridge buffer is the first CUDA device (cuda:0)
                if (device == cuda_bridge_)
                {
                    cuda_bridge_buf = buf;
                }
            }
            else if (device.is_rocm())
            {
                rocm_buffers.push_back(buf);
                // Bridge buffer is the first ROCm device (rocm:0)
                if (device == rocm_bridge_)
                {
                    rocm_bridge_buf = buf;
                }
            }
        }

        LOG_DEBUG("HeterogeneousBackend::allreduceMulti: Split into "
                  << cuda_buffers.size() << " CUDA buffers and "
                  << rocm_buffers.size() << " ROCm buffers");

        // Validate we found bridge buffers
        if (!cuda_bridge_buf)
        {
            last_error_ = "Could not find CUDA bridge buffer (device " + cuda_bridge_.toString() + ")";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }
        if (!rocm_bridge_buf)
        {
            last_error_ = "Could not find ROCm bridge buffer (device " + rocm_bridge_.toString() + ")";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // Pattern Selection via Topology Analysis
        // ═══════════════════════════════════════════════════════════════════════
        //
        // The topology analysis considers:
        // - Device counts in each domain (CUDA vs ROCm)
        // - Tensor size (large tensors benefit from reduce-scatter patterns)
        // - Symmetry (enables N-way parallel bridge transfers)
        // - GCD of device counts (for potential optimization in asymmetric cases)
        //
        size_t tensor_bytes = count * collectiveDataTypeSize(dtype);
        auto analysis = analyzeTopology(tensor_bytes);

        LOG_DEBUG("HeterogeneousBackend: Topology analysis for " << tensor_bytes << " bytes:");
        LOG_DEBUG("  CUDA=" << analysis.cuda_count << ", ROCm=" << analysis.rocm_count
                            << ", GCD=" << analysis.gcd);
        LOG_DEBUG("  Pattern: " << static_cast<int>(analysis.pattern)
                                << ", Reason: " << analysis.reason);

        // Dispatch based on selected pattern
        switch (analysis.pattern)
        {
        case AllreducePattern::SYMMETRIC_REDUCE_SCATTER:
            LOG_DEBUG("HeterogeneousBackend::allreduceMulti: Using SYMMETRIC reduce-scatter for "
                      << tensor_bytes << " bytes (" << analysis.num_chunks << "-way parallel bridge)");
            return executeReduceScatterPattern(cuda_buffers, rocm_buffers, count, dtype, op);

        case AllreducePattern::GCD_MULTI_BRIDGE:
            LOG_DEBUG("HeterogeneousBackend::allreduceMulti: Using GCD_MULTI_BRIDGE for "
                      << tensor_bytes << " bytes (" << analysis.gcd << "-way parallel bridge)");
            return executeGcdMultiBridge(cuda_buffers, rocm_buffers, count, dtype, op);

        case AllreducePattern::PARTIAL_REDUCE_SCATTER:
        {
            LOG_DEBUG("HeterogeneousBackend::allreduceMulti: Using PARTIAL reduce-scatter for "
                      << tensor_bytes << " bytes (singleton config, "
                      << analysis.num_chunks << " chunks in larger domain)");
            // Build combined buffer vector in device order for executePartialReduceScatter
            std::vector<void *> all_buffers(buffers.begin(), buffers.end());
            return executePartialReduceScatter(all_buffers, count, dtype);
        }

        case AllreducePattern::GCD_REDUCE_SCATTER:
            // Not yet implemented - fall through to standard
            LOG_DEBUG("HeterogeneousBackend::allreduceMulti: GCD reduce-scatter not implemented, "
                      << "using standard pattern");
            break;

        case AllreducePattern::STANDARD_3PHASE:
        default:
            // Fall through to standard implementation below
            break;
        }

        LOG_DEBUG("HeterogeneousBackend::allreduceMulti: Using standard 3-phase allreduce with "
                  << buffers.size() << " buffers, count=" << count);

        // ═══════════════════════════════════════════════════════════════════════
        // Standard 3-Phase Pattern
        // ═══════════════════════════════════════════════════════════════════════

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 1: Intra-domain reduce
        // ═══════════════════════════════════════════════════════════════════════
        // Reduces all CUDA buffers to cuda:0 (via NCCL if >1 device)
        // Reduces all ROCm buffers to rocm:0 (via RCCL if >1 device)
        if (!executePhase1_IntraDomainReduce(cuda_buffers, rocm_buffers, count, dtype, op))
        {
            return false;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 2: Cross-domain bridge exchange
        // ═══════════════════════════════════════════════════════════════════════
        // PCIeBAR allreduce between cuda:0 and rocm:0
        if (!executePhase2_BridgeExchange(cuda_bridge_buf, rocm_bridge_buf, count, dtype, op))
        {
            return false;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 3: Intra-domain broadcast
        // ═══════════════════════════════════════════════════════════════════════
        // Broadcasts from cuda:0 to all other CUDA devices (via NCCL if >1 device)
        // Broadcasts from rocm:0 to all other ROCm devices (via RCCL if >1 device)
        if (!executePhase3_IntraDomainBroadcast(cuda_buffers, rocm_buffers, count, dtype))
        {
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend::allreduceMulti: 3-phase allreduce complete");
        return true;
    }

    bool HeterogeneousBackend::allgather(
        const void *send_buf,
        void *recv_buf,
        size_t send_count,
        CollectiveDataType dtype)
    {
        (void)send_buf;
        (void)recv_buf;
        (void)send_count;
        (void)dtype;

        last_error_ = "HeterogeneousBackend::allgather not implemented yet";
        LOG_ERROR(last_error_);
        return false;
    }

    bool HeterogeneousBackend::allgatherv(
        const void *send_buf,
        size_t send_count,
        void *recv_buf,
        const std::vector<int> &recv_counts,
        const std::vector<int> &displacements,
        CollectiveDataType dtype)
    {
        (void)send_buf;
        (void)send_count;
        (void)recv_buf;
        (void)recv_counts;
        (void)displacements;
        (void)dtype;

        last_error_ = "HeterogeneousBackend::allgatherv not implemented yet";
        LOG_ERROR(last_error_);
        return false;
    }

    bool HeterogeneousBackend::reduceScatter(
        const void *send_buf,
        void *recv_buf,
        size_t recv_count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        (void)send_buf;
        (void)recv_buf;
        (void)recv_count;
        (void)dtype;
        (void)op;

        last_error_ = "HeterogeneousBackend::reduceScatter not implemented yet";
        LOG_ERROR(last_error_);
        return false;
    }

    bool HeterogeneousBackend::broadcast(
        void *buffer,
        size_t count,
        CollectiveDataType dtype,
        int root_rank)
    {
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)root_rank;

        last_error_ = "HeterogeneousBackend::broadcast not implemented yet";
        LOG_ERROR(last_error_);
        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 1: Intra-Domain Reduce
    // ═══════════════════════════════════════════════════════════════════════════

    HeterogeneousBackend::Phase1Plan HeterogeneousBackend::planPhase1() const
    {
        Phase1Plan plan;
        plan.nccl_device_count = cuda_devices_.size();
        plan.rccl_device_count = rocm_devices_.size();

        // NCCL reduce only needed if >1 CUDA device
        if (cuda_devices_.size() > 1 && nccl_backend_)
        {
            plan.will_call_nccl_reduce = true;
            plan.nccl_reduce_root = 0; // Bridge is always index 0 (lowest ordinal)
        }

        // RCCL reduce only needed if >1 ROCm device
        if (rocm_devices_.size() > 1 && rccl_backend_)
        {
            plan.will_call_rccl_reduce = true;
            plan.rccl_reduce_root = 0; // Bridge is always index 0 (lowest ordinal)
        }

        return plan;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Topology Analysis - Optimal Pattern Selection
    // ═══════════════════════════════════════════════════════════════════════════

    namespace
    {
        // Compute GCD using Euclidean algorithm
        size_t computeGCD(size_t a, size_t b)
        {
            while (b != 0)
            {
                size_t t = b;
                b = a % b;
                a = t;
            }
            return a;
        }
    } // anonymous namespace

    HeterogeneousBackend::TopologyAnalysis HeterogeneousBackend::analyzeTopology(size_t tensor_bytes) const
    {
        TopologyAnalysis analysis;
        analysis.cuda_count = cuda_devices_.size();
        analysis.rocm_count = rocm_devices_.size();

        // ═══════════════════════════════════════════════════════════════════
        // Step 0: Handle uninitialized/invalid state
        // ═══════════════════════════════════════════════════════════════════

        if (analysis.cuda_count == 0 || analysis.rocm_count == 0)
        {
            // Not initialized or missing a domain - return safe defaults
            analysis.is_minimal = true; // Treat as minimal to disable optimizations
            analysis.pattern = AllreducePattern::STANDARD_3PHASE;
            analysis.num_chunks = 1;
            analysis.bridge_traffic_fraction = 1;
            analysis.reason = "Invalid config (CUDA=" + std::to_string(analysis.cuda_count) +
                              ", ROCm=" + std::to_string(analysis.rocm_count) +
                              "): using safe defaults";
            return analysis;
        }

        // ═══════════════════════════════════════════════════════════════════
        // Step 1: Classify the topology
        // ═══════════════════════════════════════════════════════════════════

        analysis.is_symmetric = (analysis.cuda_count == analysis.rocm_count);
        analysis.is_minimal = (analysis.cuda_count == 1 && analysis.rocm_count == 1);
        analysis.is_cuda_singleton = (analysis.cuda_count == 1 && analysis.rocm_count > 1);
        analysis.is_rocm_singleton = (analysis.cuda_count > 1 && analysis.rocm_count == 1);
        analysis.gcd = computeGCD(analysis.cuda_count, analysis.rocm_count);

        // ═══════════════════════════════════════════════════════════════════
        // Step 2: Determine theoretical parallelism factors
        // ═══════════════════════════════════════════════════════════════════

        // Intra-domain parallelism: both NCCL and RCCL can run in parallel
        // The parallelism is limited by the larger domain's device count
        analysis.intra_domain_parallelism = static_cast<double>(
            std::max(analysis.cuda_count, analysis.rocm_count));

        // Bridge parallelism: for symmetric configs, all devices can exchange
        // in parallel (device i on CUDA ↔ device i on ROCm)
        if (analysis.is_symmetric && analysis.cuda_count > 1)
        {
            analysis.bridge_parallelism = static_cast<double>(analysis.cuda_count);
        }
        else
        {
            analysis.bridge_parallelism = 1.0; // Single bridge bottleneck
        }

        // ═══════════════════════════════════════════════════════════════════
        // Step 3: Select optimal pattern based on topology and tensor size
        // ═══════════════════════════════════════════════════════════════════

        bool is_large_tensor = (tensor_bytes >= REDUCE_SCATTER_THRESHOLD);

        // Helper to compute pipelining eligibility before returning
        auto computePipelining = [&]()
        {
            if (tensor_bytes >= PIPELINE_MIN_TENSOR_SIZE)
            {
                analysis.pipelining_eligible = true;
                analysis.pipeline_chunks = (tensor_bytes + PIPELINE_CHUNK_SIZE - 1) / PIPELINE_CHUNK_SIZE;
                // At least 2 chunks needed for meaningful pipelining
                if (analysis.pipeline_chunks < 2)
                {
                    analysis.pipelining_eligible = false;
                    analysis.pipeline_chunks = 0;
                }
            }
        };

        // Case 1: Minimal config (1+1) - no intra-domain parallelism possible
        if (analysis.is_minimal)
        {
            analysis.pattern = AllreducePattern::STANDARD_3PHASE;
            analysis.num_chunks = 1;
            analysis.bridge_traffic_fraction = 1;
            analysis.reason = "Minimal 1+1 config: no intra-domain parallelism benefit";
            computePipelining();
            return analysis;
        }

        // Case 2: Small tensor - latency dominates, use simpler pattern
        if (!is_large_tensor)
        {
            analysis.pattern = AllreducePattern::STANDARD_3PHASE;
            analysis.num_chunks = 1;
            analysis.bridge_traffic_fraction = 1;
            analysis.reason = "Small tensor (" + std::to_string(tensor_bytes) +
                              " bytes < " + std::to_string(REDUCE_SCATTER_THRESHOLD) +
                              "): latency-optimized standard pattern";
            // No pipelining for small tensors
            return analysis;
        }

        // Case 3: Symmetric config with large tensor - full reduce-scatter
        if (analysis.is_symmetric)
        {
            analysis.pattern = AllreducePattern::SYMMETRIC_REDUCE_SCATTER;
            analysis.num_chunks = analysis.cuda_count;
            analysis.bridge_traffic_fraction = analysis.cuda_count;
            analysis.reason = "Symmetric " + std::to_string(analysis.cuda_count) + "+" +
                              std::to_string(analysis.rocm_count) +
                              " config: " + std::to_string(analysis.cuda_count) +
                              "-way parallel bridge (1/" + std::to_string(analysis.cuda_count) +
                              " traffic per transfer)";
            computePipelining();
            return analysis;
        }

        // Case 4: Asymmetric config - evaluate options
        // ═══════════════════════════════════════════════════════════════════
        // For asymmetric configs, we have several options:
        //
        // A) STANDARD_3PHASE: Simple, works everywhere
        //    - Bridge traffic: 100% of tensor
        //    - Intra-domain: Reduce/Broadcast parallelize within each domain
        //
        // B) PARTIAL_REDUCE_SCATTER: RS in larger domain only
        //    - Requires p2p staging for non-bridge devices
        //    - Bridge traffic: Still 100% (bridge is bottleneck)
        //    - But: Reduced memory pressure within larger domain
        //
        // C) GCD_REDUCE_SCATTER: GCD-based chunking
        //    - If GCD > 1, can do GCD-way parallel bridge
        //    - Example: 2 CUDA + 4 ROCm → GCD=2 → 2-way parallel
        //    - Requires p2p staging
        //
        // Currently, only STANDARD_3PHASE is fully implemented.
        // PARTIAL and GCD patterns require ICollectiveBackend::sendrecv().
        // ═══════════════════════════════════════════════════════════════════

        // Check if GCD-based optimization would help
        if (analysis.gcd > 1)
        {
            // GCD > 1 means we can do GCD-way parallel bridge
            // Example: 2 CUDA + 4 ROCm → 2 parallel bridges (CUDA[0]↔ROCm[0], CUDA[1]↔ROCm[2])
            analysis.pattern = AllreducePattern::GCD_MULTI_BRIDGE;
            analysis.num_chunks = analysis.gcd;
            analysis.bridge_traffic_fraction = analysis.gcd; // GCD parallel bridges
            analysis.bridge_parallelism = static_cast<double>(analysis.gcd);
            analysis.reason = "Asymmetric " + std::to_string(analysis.cuda_count) + "+" +
                              std::to_string(analysis.rocm_count) +
                              " using GCD=" + std::to_string(analysis.gcd) +
                              "-way parallel bridge";

            LOG_DEBUG("HeterogeneousBackend: Using GCD multi-bridge pattern "
                      << "(GCD=" << analysis.gcd << " parallel bridges)");
            computePipelining();
            return analysis;
        }

        // Singleton cases: one domain has 1 device, other has more
        if (analysis.is_cuda_singleton || analysis.is_rocm_singleton)
        {
            // The larger domain benefits from reduce-scatter internally.
            // Bridge is still a bottleneck, but partial RS reduces memory
            // pressure in the larger domain during the operation.
            size_t larger = std::max(analysis.cuda_count, analysis.rocm_count);

            analysis.pattern = AllreducePattern::PARTIAL_REDUCE_SCATTER;
            analysis.num_chunks = larger;         // Chunks from the larger domain
            analysis.bridge_traffic_fraction = 1; // Bridge still processes all data (serial)

            // Build descriptive reason string
            std::string config_str;
            if (analysis.is_cuda_singleton)
            {
                config_str = "1+" + std::to_string(larger); // 1 CUDA + N ROCm
            }
            else
            {
                config_str = std::to_string(larger) + "+1"; // N CUDA + 1 ROCm
            }
            analysis.reason = "Singleton " + config_str +
                              ": partial RS reduces memory in larger domain";

            LOG_DEBUG("HeterogeneousBackend: Singleton config (" << config_str
                                                                 << ") using partial reduce-scatter in larger domain");
            computePipelining();
            return analysis;
        }

        // General asymmetric case with GCD=1 (e.g., 2 CUDA + 3 ROCm)
        analysis.pattern = AllreducePattern::STANDARD_3PHASE;
        analysis.num_chunks = 1;
        analysis.bridge_traffic_fraction = 1;
        analysis.reason = "Asymmetric " + std::to_string(analysis.cuda_count) + "+" +
                          std::to_string(analysis.rocm_count) +
                          " config with GCD=1: standard pattern (no parallel bridge benefit)";

        // ═══════════════════════════════════════════════════════════════════
        // Step 4: Determine pipelining eligibility (for all patterns)
        // ═══════════════════════════════════════════════════════════════════
        // Pipelining can overlap bridge transfers with intra-domain operations.
        // Only beneficial for large enough tensors that can be chunked.
        if (tensor_bytes >= PIPELINE_MIN_TENSOR_SIZE)
        {
            analysis.pipelining_eligible = true;
            // Calculate number of pipeline chunks based on tensor size and chunk size
            analysis.pipeline_chunks = (tensor_bytes + PIPELINE_CHUNK_SIZE - 1) / PIPELINE_CHUNK_SIZE;
            // At least 2 chunks needed for meaningful pipelining
            if (analysis.pipeline_chunks < 2)
            {
                analysis.pipelining_eligible = false;
                analysis.pipeline_chunks = 0;
            }
        }

        return analysis;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GCD Multi-Bridge Pattern
    // ═══════════════════════════════════════════════════════════════════════════

    std::vector<DevicePair> HeterogeneousBackend::computeBridgePairs() const
    {
        std::vector<DevicePair> pairs;

        if (cuda_devices_.empty() || rocm_devices_.empty())
        {
            return pairs;
        }

        // G = GCD(cuda_count, rocm_count)
        size_t G = computeGCD(cuda_devices_.size(), rocm_devices_.size());

        if (G == 0)
        {
            return pairs;
        }

        // For i in 0..G-1:
        //   cuda_idx = i
        //   rocm_idx = i * (rocm_count / G)
        //   Create pair: {cuda_devices_[cuda_idx], rocm_devices_[rocm_idx], i}
        size_t rocm_stride = rocm_devices_.size() / G;

        pairs.reserve(G);
        for (size_t i = 0; i < G; ++i)
        {
            size_t cuda_idx = i;
            size_t rocm_idx = i * rocm_stride;

            DevicePair pair;
            pair.cuda_device = cuda_devices_[cuda_idx];
            pair.rocm_device = rocm_devices_[rocm_idx];
            pair.pair_index = static_cast<int>(i);

            pairs.push_back(pair);

            LOG_DEBUG("HeterogeneousBackend: Bridge pair " << i << ": "
                                                           << pair.cuda_device.toString() << " <-> "
                                                           << pair.rocm_device.toString());
        }

        return pairs;
    }

    bool HeterogeneousBackend::executeGcdMultiBridge(
        const std::vector<void *> &cuda_buffers,
        const std::vector<void *> &rocm_buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        size_t G = computeGCD(cuda_devices_.size(), rocm_devices_.size());
        if (G <= 1)
        {
            // Fall back to standard pattern if GCD <= 1
            LOG_DEBUG("HeterogeneousBackend: GCD <= 1, using standard 3-phase pattern");
            return executeStandard3PhaseAllreduce(cuda_buffers, rocm_buffers, count, dtype, op);
        }

        LOG_DEBUG("HeterogeneousBackend: Executing GCD_MULTI_BRIDGE pattern (GCD=" << G << ")");
        LOG_DEBUG("  CUDA devices: " << cuda_devices_.size() << ", ROCm devices: " << rocm_devices_.size());

        // Compute bridge pairs if not already cached
        if (bridge_pairs_.empty() || bridge_pairs_.size() != G)
        {
            bridge_pairs_ = computeBridgePairs();
        }

        if (bridge_pairs_.size() != G)
        {
            last_error_ = "Failed to compute bridge pairs (expected " + std::to_string(G) +
                          ", got " + std::to_string(bridge_pairs_.size()) + ")";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 1: Parallel intra-domain ALLREDUCE (not reduce!)
        // ═══════════════════════════════════════════════════════════════════════
        // Unlike the standard 3-phase pattern which uses reduce, we use allreduce
        // so that ALL devices in each domain have the domain's result.
        // This is necessary because multiple devices participate in bridge exchange.

        bool nccl_success = true;
        bool rccl_success = true;
        std::string nccl_error;
        std::string rccl_error;

        std::thread nccl_thread;
        std::thread rccl_thread;

        // NCCL allreduce (if >1 CUDA device)
        if (cuda_devices_.size() > 1 && nccl_backend_)
        {
            LOG_DEBUG("HeterogeneousBackend GCD Phase1: NCCL allreduce with "
                      << cuda_devices_.size() << " devices");

            nccl_thread = std::thread([&]()
                                      {
                std::vector<void*> mutable_buffers(cuda_buffers.begin(), cuda_buffers.end());
                nccl_success = nccl_backend_->allreduceMulti(mutable_buffers, count, dtype, op);
                if (!nccl_success) {
                    nccl_error = nccl_backend_->lastError();
                } });
        }

        // RCCL allreduce (if >1 ROCm device)
        if (rocm_devices_.size() > 1 && rccl_backend_)
        {
            LOG_DEBUG("HeterogeneousBackend GCD Phase1: RCCL allreduce with "
                      << rocm_devices_.size() << " devices");

            rccl_thread = std::thread([&]()
                                      {
                std::vector<void*> mutable_buffers(rocm_buffers.begin(), rocm_buffers.end());
                rccl_success = rccl_backend_->allreduceMulti(mutable_buffers, count, dtype, op);
                if (!rccl_success) {
                    rccl_error = rccl_backend_->lastError();
                } });
        }

        if (nccl_thread.joinable())
            nccl_thread.join();
        if (rccl_thread.joinable())
            rccl_thread.join();

        if (!nccl_success)
        {
            last_error_ = "GCD Phase 1: NCCL allreduce failed: " + nccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }
        if (!rccl_success)
        {
            last_error_ = "GCD Phase 1: RCCL allreduce failed: " + rccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend GCD Phase1: Intra-domain allreduce complete");

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 2: Multi-pair PCIeBAR allreduce
        // ═══════════════════════════════════════════════════════════════════════
        // Each bridge pair exchanges and reduces in parallel.

        LOG_DEBUG("HeterogeneousBackend GCD Phase2: Multi-pair bridge exchange ("
                  << G << " pairs)");

        // Initialize multi-pair bridge backend if needed
        if (!multi_pair_bridge_backend_ || !multi_pair_bridge_backend_->isMultiPairMode())
        {
            multi_pair_bridge_backend_ = std::make_unique<PCIeBARBackend>();
            if (!multi_pair_bridge_backend_->initializeMultiPair(bridge_pairs_))
            {
                last_error_ = "GCD Phase 2: Failed to initialize multi-pair bridge backend";
                LOG_ERROR("HeterogeneousBackend: " << last_error_);
                multi_pair_bridge_backend_.reset();
                // Fall back to serial bridge exchanges using standard bridge_backend_
                LOG_WARN("HeterogeneousBackend: Falling back to serial bridge exchanges");
            }
        }

        // Build buffer vectors for each pair
        // cuda_buffers[pair.cuda_device.ordinal] and rocm_buffers[corresponding_idx]
        std::vector<void *> cuda_pair_buffers;
        std::vector<void *> rocm_pair_buffers;
        cuda_pair_buffers.reserve(G);
        rocm_pair_buffers.reserve(G);

        // Map device ordinals to buffer indices
        // cuda_buffers are in cuda_devices_ order (sorted by ordinal)
        // rocm_buffers are in rocm_devices_ order (sorted by ordinal)
        for (const auto &pair : bridge_pairs_)
        {
            // Find cuda buffer index
            size_t cuda_buf_idx = 0;
            for (size_t i = 0; i < cuda_devices_.size(); ++i)
            {
                if (cuda_devices_[i] == pair.cuda_device)
                {
                    cuda_buf_idx = i;
                    break;
                }
            }

            // Find rocm buffer index
            size_t rocm_buf_idx = 0;
            for (size_t i = 0; i < rocm_devices_.size(); ++i)
            {
                if (rocm_devices_[i] == pair.rocm_device)
                {
                    rocm_buf_idx = i;
                    break;
                }
            }

            cuda_pair_buffers.push_back(cuda_buffers[cuda_buf_idx]);
            rocm_pair_buffers.push_back(rocm_buffers[rocm_buf_idx]);

            LOG_DEBUG("  Pair " << pair.pair_index << ": CUDA buf[" << cuda_buf_idx
                                << "] <-> ROCm buf[" << rocm_buf_idx << "]");
        }

        // Execute multi-pair allreduce
        if (multi_pair_bridge_backend_ && multi_pair_bridge_backend_->isMultiPairMode())
        {
            if (!multi_pair_bridge_backend_->allreduceMultiPair(
                    cuda_pair_buffers, rocm_pair_buffers, count, dtype))
            {
                last_error_ = "GCD Phase 2: Multi-pair bridge allreduce failed";
                LOG_ERROR("HeterogeneousBackend: " << last_error_);
                return false;
            }
        }
        else
        {
            // Fallback: Serial bridge exchanges using standard bridge_backend_
            // This is slower but works when multi-pair init fails
            for (size_t i = 0; i < G; ++i)
            {
                LOG_DEBUG("  Serial bridge exchange for pair " << i);
                if (!executePhase2_BridgeExchange(
                        cuda_pair_buffers[i], rocm_pair_buffers[i], count, dtype, op))
                {
                    last_error_ = "GCD Phase 2: Serial bridge exchange failed for pair " +
                                  std::to_string(i);
                    LOG_ERROR("HeterogeneousBackend: " << last_error_);
                    return false;
                }
            }
        }

        LOG_DEBUG("HeterogeneousBackend GCD Phase2: Bridge exchange complete");

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 3: Intra-domain broadcast from bridge devices
        // ═══════════════════════════════════════════════════════════════════════
        // After Phase 2, each bridge device has the global result.
        // - CUDA: All CUDA devices already have the result (from Phase 1 allreduce +
        //         bridge update). Actually, only bridge CUDAs have the cross-domain
        //         result. Non-bridge CUDAs need update.
        // - ROCm: Bridge ROCm devices have result. Need to broadcast to their groups.

        LOG_DEBUG("HeterogeneousBackend GCD Phase3: Intra-domain distribution");

        // For CUDA: If >1 device and not all are bridge devices, we need a broadcast
        // from bridge CUDAs to non-bridge CUDAs.
        // With GCD-way bridging: CUDA[0..G-1] are bridges, CUDA[G..N-1] need update
        if (cuda_devices_.size() > G && nccl_backend_)
        {
            // CUDA bridge devices have the global result; need to update others
            // Use NCCL allreduce with the bridge results to propagate
            // Actually, since all CUDA had the same value before Phase 2 (from Phase 1 allreduce),
            // and we updated bridge CUDAs with the cross-domain sum, we need to copy
            // from a bridge CUDA to all non-bridge CUDAs.

            // Simplest approach: broadcast from CUDA[0] to all CUDA devices
            LOG_DEBUG("HeterogeneousBackend GCD Phase3: NCCL broadcast from bridge to all CUDA");

            nccl_thread = std::thread([&]()
                                      {
                std::vector<void*> mutable_buffers(cuda_buffers.begin(), cuda_buffers.end());
                nccl_success = nccl_backend_->broadcastMulti(mutable_buffers, count, dtype, 0);
                if (!nccl_success) {
                    nccl_error = nccl_backend_->lastError();
                } });
        }
        else
        {
            // All CUDA devices are bridges or only 1 CUDA device - no broadcast needed
            nccl_success = true;
        }

        // For ROCm: Bridge devices at indices 0, stride, 2*stride, ... have the result
        // Each bridge needs to broadcast to its "group" of rocm_devices_.size()/G devices
        if (rocm_devices_.size() > G && rccl_backend_)
        {
            // Complex case: need to broadcast from each bridge ROCm to its group
            // For now, use a simple full broadcast from ROCm[0]
            // TODO: Optimize with multiple small broadcasts for better parallelism

            LOG_DEBUG("HeterogeneousBackend GCD Phase3: RCCL broadcast from bridge to all ROCm");

            rccl_thread = std::thread([&]()
                                      {
                std::vector<void*> mutable_buffers(rocm_buffers.begin(), rocm_buffers.end());
                rccl_success = rccl_backend_->broadcastMulti(mutable_buffers, count, dtype, 0);
                if (!rccl_success) {
                    rccl_error = rccl_backend_->lastError();
                } });
        }
        else
        {
            // All ROCm devices are bridges or only 1 ROCm device - no broadcast needed
            rccl_success = true;
        }

        if (nccl_thread.joinable())
            nccl_thread.join();
        if (rccl_thread.joinable())
            rccl_thread.join();

        if (!nccl_success)
        {
            last_error_ = "GCD Phase 3: NCCL broadcast failed: " + nccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }
        if (!rccl_success)
        {
            last_error_ = "GCD Phase 3: RCCL broadcast failed: " + rccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend GCD Phase3: Complete");
        LOG_DEBUG("HeterogeneousBackend: GCD_MULTI_BRIDGE allreduce complete");
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Reduce-Scatter Pattern (for large tensors)
    // ═══════════════════════════════════════════════════════════════════════════

    bool HeterogeneousBackend::shouldUseReduceScatterPattern(size_t tensor_bytes) const
    {
        // Check threshold
        if (tensor_bytes < REDUCE_SCATTER_THRESHOLD)
        {
            return false;
        }

        // ═══════════════════════════════════════════════════════════════════
        // CRITICAL: Reduce-scatter pattern REQUIRES symmetric device counts
        // ═══════════════════════════════════════════════════════════════════
        // The reduce-scatter pattern partitions data into chunks based on
        // device count in each domain. When counts differ, the partitioning
        // doesn't align:
        //   - 1 CUDA: chunk = full tensor
        //   - 2 ROCm: chunk = half tensor
        // Bridge exchange with min(full, half) = half leaves data unreduced.
        //
        // TODO: Future work could implement asymmetric reduce-scatter that
        // properly handles mismatched device counts by using GCD-based
        // partitioning or multiple bridge exchanges.
        //
        // For now, fall back to standard 3-phase pattern for asymmetric cases.
        if (cuda_devices_.size() != rocm_devices_.size())
        {
            LOG_DEBUG("HeterogeneousBackend: Skipping reduce-scatter pattern for asymmetric "
                      << "device counts (CUDA=" << cuda_devices_.size()
                      << ", ROCm=" << rocm_devices_.size() << ")");
            return false;
        }

        // Need at least one domain with >1 device to benefit from reduce-scatter
        // Otherwise, the chunk sizes are the same as full tensor sizes
        return (cuda_devices_.size() > 1 || rocm_devices_.size() > 1);
    }

    bool HeterogeneousBackend::shouldUseAdaptiveAsymmetricPattern(size_t tensor_bytes) const
    {
        // Check threshold
        if (tensor_bytes < REDUCE_SCATTER_THRESHOLD)
        {
            return false;
        }

        // Must have asymmetric device counts
        if (cuda_devices_.size() == rocm_devices_.size())
        {
            return false;
        }

        // Need at least one domain with >1 device (the "larger" domain)
        // The pattern chunks based on the larger domain, so we need >1 to benefit
        size_t larger_count = std::max(cuda_devices_.size(), rocm_devices_.size());

        // ═══════════════════════════════════════════════════════════════════
        // IMPLEMENTATION STATUS:
        //
        // The adaptive asymmetric pattern requires intra-domain staging to
        // relay chunks from non-bridge devices through the bridge. This needs
        // either:
        //   1. Point-to-point send/recv primitives (ncclSend/ncclRecv), or
        //   2. Direct GPU peer memcpy (hipMemcpyPeer/cudaMemcpyPeer), or
        //   3. Host-mediated staging (slow, defeats the purpose)
        //
        // Currently the ICollectiveBackend interface doesn't expose p2p
        // primitives, so we cannot implement efficient intra-domain staging.
        //
        // For now, the adaptive pattern is DISABLED. The standard 3-phase
        // pattern handles asymmetric configurations correctly, though not
        // optimally for bandwidth.
        //
        // TODO(future): Add ICollectiveBackend::sendrecv() primitive and
        // implement proper intra-domain staging for adaptive pattern.
        // ═══════════════════════════════════════════════════════════════════

        if (larger_count > 1)
        {
            LOG_DEBUG("HeterogeneousBackend: Adaptive asymmetric pattern requires p2p staging "
                      << "for " << larger_count << " devices (not yet implemented)");
            return false;
        }

        // With larger_count <= 1, either:
        // - Both domains have 1 device (symmetric, use standard pattern)
        // - Smaller domain has 0 devices (degenerate case)
        // Neither case benefits from the adaptive pattern
        return false;
    }

    HeterogeneousBackend::AsymmetricReduceScatterPlan HeterogeneousBackend::planAsymmetricReduceScatter(
        size_t count, size_t element_size) const
    {
        AsymmetricReduceScatterPlan plan;
        plan.cuda_device_count = cuda_devices_.size();
        plan.rocm_device_count = rocm_devices_.size();

        size_t tensor_bytes = count * element_size;
        plan.use_adaptive_pattern = shouldUseAdaptiveAsymmetricPattern(tensor_bytes);

        if (!plan.use_adaptive_pattern)
        {
            // Not using adaptive pattern - return defaults
            plan.num_chunks = 1;
            plan.chunk_elements = count;
            plan.last_chunk_elements = count;
            return plan;
        }

        // Determine which domain is larger
        plan.cuda_is_larger = (cuda_devices_.size() > rocm_devices_.size());
        plan.larger_domain_count = std::max(cuda_devices_.size(), rocm_devices_.size());
        plan.smaller_domain_count = std::min(cuda_devices_.size(), rocm_devices_.size());

        // Number of chunks = larger domain device count
        // This allows the larger domain to do a proper reduce-scatter
        plan.num_chunks = plan.larger_domain_count;

        // Elements per chunk (integer division)
        plan.chunk_elements = count / plan.num_chunks;

        // Last chunk may have remainder
        size_t full_chunks = count / plan.chunk_elements;
        size_t remainder = count % plan.chunk_elements;
        if (remainder > 0 && full_chunks < plan.num_chunks)
        {
            plan.last_chunk_elements = remainder;
        }
        else
        {
            plan.last_chunk_elements = plan.chunk_elements;
        }

        return plan;
    }

    HeterogeneousBackend::ReduceScatterPlan HeterogeneousBackend::planReduceScatter(
        size_t count, size_t element_size) const
    {
        ReduceScatterPlan plan;
        plan.cuda_device_count = cuda_devices_.size();
        plan.rocm_device_count = rocm_devices_.size();

        size_t tensor_bytes = count * element_size;
        plan.use_reduce_scatter_pattern = shouldUseReduceScatterPattern(tensor_bytes);

        if (!plan.use_reduce_scatter_pattern)
        {
            // Standard pattern: full tensor exchange
            plan.cuda_chunk_count = count;
            plan.rocm_chunk_count = count;
            plan.bridge_exchange_count = count;
            return plan;
        }

        // Compute chunk sizes for each domain
        // After reduce-scatter, each device in a domain has count/N elements
        size_t cuda_count = cuda_devices_.size();
        size_t rocm_count = rocm_devices_.size();

        // Integer division with remainder going to last device
        plan.cuda_chunk_count = (cuda_count > 0) ? (count / cuda_count) : 0;
        plan.rocm_chunk_count = (rocm_count > 0) ? (count / rocm_count) : 0;

        // Bridge exchange transfers the smaller of the two chunks
        // This is where the bandwidth savings come from
        plan.bridge_exchange_count = std::min(plan.cuda_chunk_count, plan.rocm_chunk_count);

        return plan;
    }

    bool HeterogeneousBackend::executeReduceScatterPattern(
        const std::vector<void *> &cuda_buffers,
        const std::vector<void *> &rocm_buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        // Reduce-scatter pattern only works with symmetric device counts
        // (same number of devices in each domain). For asymmetric configs,
        // use the standard 3-phase pattern.
        if (cuda_devices_.size() != rocm_devices_.size())
        {
            LOG_DEBUG("HeterogeneousBackend: Asymmetric device counts (CUDA="
                      << cuda_devices_.size() << ", ROCm=" << rocm_devices_.size()
                      << "), using standard 3-phase pattern instead of reduce-scatter");
            return executeStandard3PhaseAllreduce(cuda_buffers, rocm_buffers, count, dtype, op);
        }

        LOG_DEBUG("HeterogeneousBackend: Using reduce-scatter pattern for " << count << " elements");

        // Compute chunk sizes
        size_t cuda_chunk_count = count / cuda_devices_.size();
        size_t rocm_chunk_count = count / rocm_devices_.size();

        // ═══════════════════════════════════════════════════════════════════
        // Phase 1: Reduce-Scatter in each domain
        // ═══════════════════════════════════════════════════════════════════
        // After this phase, each device has a chunk of the reduced tensor:
        // - cuda_buffers[i] has elements [i*cuda_chunk_count, (i+1)*cuda_chunk_count)
        // - rocm_buffers[i] has elements [i*rocm_chunk_count, (i+1)*rocm_chunk_count)

        bool nccl_success = true;
        bool rccl_success = true;
        std::string nccl_error;
        std::string rccl_error;

        std::thread nccl_thread;
        std::thread rccl_thread;

        // NCCL reduce-scatter (if >1 CUDA device)
        if (cuda_devices_.size() > 1 && nccl_backend_)
        {
            LOG_DEBUG("HeterogeneousBackend RS Phase1: NCCL reduce-scatter with "
                      << cuda_devices_.size() << " devices, chunk=" << cuda_chunk_count);

            nccl_thread = std::thread([&]()
                                      {
                // For reduce-scatter: send_buffers = recv_buffers (in-place conceptually)
                // But NCCL wants separate send and recv pointers
                std::vector<const void*> send_bufs(cuda_buffers.begin(), cuda_buffers.end());
                nccl_success = nccl_backend_->reduceScatterMulti(
                    send_bufs, cuda_buffers, cuda_chunk_count, dtype, op);
                if (!nccl_success) {
                    nccl_error = nccl_backend_->lastError();
                } });
        }

        // RCCL reduce-scatter (if >1 ROCm device)
        if (rocm_devices_.size() > 1 && rccl_backend_)
        {
            LOG_DEBUG("HeterogeneousBackend RS Phase1: RCCL reduce-scatter with "
                      << rocm_devices_.size() << " devices, chunk=" << rocm_chunk_count);

            rccl_thread = std::thread([&]()
                                      {
                std::vector<const void*> send_bufs(rocm_buffers.begin(), rocm_buffers.end());
                rccl_success = rccl_backend_->reduceScatterMulti(
                    send_bufs, rocm_buffers, rocm_chunk_count, dtype, op);
                if (!rccl_success) {
                    rccl_error = rccl_backend_->lastError();
                } });
        }

        if (nccl_thread.joinable())
            nccl_thread.join();
        if (rccl_thread.joinable())
            rccl_thread.join();

        if (!nccl_success)
        {
            last_error_ = "NCCL reduce-scatter failed: " + nccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }
        if (!rccl_success)
        {
            last_error_ = "RCCL reduce-scatter failed: " + rccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend RS Phase1: Reduce-scatter complete");

        // ═══════════════════════════════════════════════════════════════════
        // Phase 2: Bridge exchange (only the bridge device chunks!)
        // ═══════════════════════════════════════════════════════════════════
        // Exchange cuda_buffers[0] chunk with rocm_buffers[0] chunk
        // This is MUCH smaller than the full tensor - the key optimization!

        void *cuda_bridge_buf = cuda_buffers[0];
        void *rocm_bridge_buf = rocm_buffers[0];

        // The bridge exchange count is the minimum chunk size
        // Each bridge device contributes its chunk for the cross-domain allreduce
        size_t bridge_exchange_count = std::min(cuda_chunk_count, rocm_chunk_count);

        LOG_DEBUG("HeterogeneousBackend RS Phase2: Bridge exchange "
                  << bridge_exchange_count << " elements (vs " << count << " full tensor)");

        if (!bridge_backend_->allreduce(cuda_bridge_buf, bridge_exchange_count, dtype, op))
        {
            last_error_ = "Phase 2: Bridge allreduce failed: " + bridge_backend_->lastError();
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend RS Phase2: Bridge exchange complete");

        // ═══════════════════════════════════════════════════════════════════
        // Phase 3: AllGather in each domain
        // ═══════════════════════════════════════════════════════════════════
        // Reassemble full reduced tensor on all devices

        nccl_success = true;
        rccl_success = true;

        // NCCL allgather (if >1 CUDA device)
        if (cuda_devices_.size() > 1 && nccl_backend_)
        {
            LOG_DEBUG("HeterogeneousBackend RS Phase3: NCCL allgather");

            nccl_thread = std::thread([&]()
                                      {
                // Each device sends its chunk, receives full tensor
                std::vector<const void*> send_bufs(cuda_buffers.begin(), cuda_buffers.end());
                nccl_success = nccl_backend_->allgatherMulti(
                    send_bufs, cuda_buffers, cuda_chunk_count, dtype);
                if (!nccl_success) {
                    nccl_error = nccl_backend_->lastError();
                } });
        }

        // RCCL allgather (if >1 ROCm device)
        if (rocm_devices_.size() > 1 && rccl_backend_)
        {
            LOG_DEBUG("HeterogeneousBackend RS Phase3: RCCL allgather");

            rccl_thread = std::thread([&]()
                                      {
                std::vector<const void*> send_bufs(rocm_buffers.begin(), rocm_buffers.end());
                rccl_success = rccl_backend_->allgatherMulti(
                    send_bufs, rocm_buffers, rocm_chunk_count, dtype);
                if (!rccl_success) {
                    rccl_error = rccl_backend_->lastError();
                } });
        }

        if (nccl_thread.joinable())
            nccl_thread.join();
        if (rccl_thread.joinable())
            rccl_thread.join();

        if (!nccl_success)
        {
            last_error_ = "NCCL allgather failed: " + nccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }
        if (!rccl_success)
        {
            last_error_ = "RCCL allgather failed: " + rccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend RS Phase3: AllGather complete");
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Adaptive Asymmetric Reduce-Scatter Pattern
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // For asymmetric configs (e.g., 1 CUDA + 2 ROCm), standard reduce-scatter
    // doesn't work because chunk sizes don't align. This adaptive pattern:
    //
    // 1. Chunks based on the LARGER domain (e.g., 2 chunks for 2 ROCm devices)
    // 2. Larger domain does reduce-scatter (each device gets 1 chunk)
    // 3. Smaller domain exchanges per-chunk with the larger domain's devices
    //
    // Example: 1 CUDA + 2 ROCm, tensor [A, B] (2 chunks)
    //   Initial:   CUDA=[A,B]   ROCm0=[A,B]   ROCm1=[A,B]
    //
    //   Phase 1:   (ROCm reduce-scatter)
    //              CUDA=[A,B]   ROCm0=A_reduced   ROCm1=B_reduced
    //
    //   Phase 2:   (Chunked bridge exchange)
    //              - CUDA chunk A ↔ ROCm0 chunk A → both have A_global
    //              - CUDA chunk B ↔ ROCm1 chunk B → both have B_global
    //              CUDA=[A_global, B_global]   ROCm0=A_global   ROCm1=B_global
    //
    //   Phase 3:   (ROCm allgather)
    //              CUDA=[A_global, B_global]   ROCm0=[A_global, B_global]   ROCm1=[A_global, B_global]

    bool HeterogeneousBackend::executeAdaptiveAsymmetricReduceScatter(
        const std::vector<void *> &cuda_buffers,
        const std::vector<void *> &rocm_buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        // Get element size for pointer arithmetic
        size_t element_size = collectiveDataTypeSize(dtype);

        // Plan the adaptive pattern
        auto plan = planAsymmetricReduceScatter(count, element_size);

        if (!plan.use_adaptive_pattern)
        {
            // Shouldn't happen if caller checked shouldUseAdaptiveAsymmetricPattern
            LOG_WARN("HeterogeneousBackend: Adaptive pattern not applicable, using standard 3-phase");
            return executeStandard3PhaseAllreduce(cuda_buffers, rocm_buffers, count, dtype, op);
        }

        LOG_DEBUG("HeterogeneousBackend: Using ADAPTIVE asymmetric reduce-scatter pattern");
        LOG_DEBUG("  CUDA devices: " << plan.cuda_device_count << ", ROCm devices: " << plan.rocm_device_count);
        LOG_DEBUG("  Larger domain: " << (plan.cuda_is_larger ? "CUDA" : "ROCm")
                                      << " (" << plan.larger_domain_count << " devices)");
        LOG_DEBUG("  Chunks: " << plan.num_chunks << ", elements per chunk: " << plan.chunk_elements);

        // References for clarity
        const auto &larger_buffers = plan.cuda_is_larger ? cuda_buffers : rocm_buffers;
        const auto &smaller_buffers = plan.cuda_is_larger ? rocm_buffers : cuda_buffers;
        const auto &larger_devices = plan.cuda_is_larger ? cuda_devices_ : rocm_devices_;
        const auto &smaller_devices = plan.cuda_is_larger ? rocm_devices_ : cuda_devices_;

        // Select the appropriate backend for the larger domain
        // Note: NCCL and RCCL have compatible interfaces via ICollectiveBackend
        ICollectiveBackend *larger_backend =
            plan.cuda_is_larger
                ? static_cast<ICollectiveBackend *>(nccl_backend_.get())
                : static_cast<ICollectiveBackend *>(rccl_backend_.get());

        // ═══════════════════════════════════════════════════════════════════
        // Phase 1: Reduce-scatter in the LARGER domain
        // ═══════════════════════════════════════════════════════════════════
        // The larger domain does reduce-scatter: each device ends up with 1 chunk.
        // The smaller domain has already "reduced" its contribution (single device
        // or will be done via bridge).

        bool larger_success = true;
        std::string larger_error;

        if (plan.larger_domain_count > 1 && larger_backend)
        {
            LOG_DEBUG("HeterogeneousBackend Adaptive Phase1: "
                      << (plan.cuda_is_larger ? "NCCL" : "RCCL")
                      << " reduce-scatter with " << plan.larger_domain_count << " devices");

            std::vector<const void *> send_bufs(larger_buffers.begin(), larger_buffers.end());
            larger_success = larger_backend->reduceScatterMulti(
                send_bufs, const_cast<std::vector<void *> &>(larger_buffers),
                plan.chunk_elements, dtype, op);

            if (!larger_success)
            {
                larger_error = larger_backend->lastError();
            }
        }

        if (!larger_success)
        {
            last_error_ = "Adaptive Phase 1: Larger domain reduce-scatter failed: " + larger_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Adaptive Phase1: Complete");

        // ═══════════════════════════════════════════════════════════════════
        // Phase 2: Chunked bridge exchange
        // ═══════════════════════════════════════════════════════════════════
        // Exchange each chunk between smaller domain and corresponding larger domain device.
        //
        // For 1 CUDA + 2 ROCm (CUDA is smaller):
        //   - Chunk 0: CUDA[0..chunk_elements] ↔ ROCm0[0..chunk_elements]
        //   - Chunk 1: CUDA[chunk_elements..2*chunk_elements] ↔ ROCm1[0..chunk_elements]
        //
        // After reduce-scatter, larger_buffers[i] starts at offset 0 but contains only
        // chunk_elements of data (the reduced chunk i). We need to exchange:
        //   smaller_buffers[0] + chunk_offset ↔ larger_buffers[chunk_idx]
        //
        // The smaller domain's single buffer contains the full tensor, so we slice it.
        // The larger domain's each buffer contains only its chunk (at offset 0).

        LOG_DEBUG("HeterogeneousBackend Adaptive Phase2: Chunked bridge exchange");

        // For the smaller domain (e.g., 1 CUDA device), we need to iterate through chunks
        // and exchange with each device in the larger domain
        void *smaller_buffer = smaller_buffers[0]; // Single device in smaller domain

        for (size_t chunk_idx = 0; chunk_idx < plan.num_chunks; ++chunk_idx)
        {
            // Calculate chunk size (last chunk may be different)
            size_t this_chunk_elements = (chunk_idx == plan.num_chunks - 1)
                                             ? plan.last_chunk_elements
                                             : plan.chunk_elements;

            // Calculate offset into smaller domain's buffer for this chunk
            size_t chunk_offset_elements = chunk_idx * plan.chunk_elements;
            void *smaller_chunk_ptr = static_cast<char *>(smaller_buffer) +
                                      chunk_offset_elements * element_size;

            // Larger domain device for this chunk (after reduce-scatter, each device has
            // its chunk at offset 0)
            void *larger_chunk_ptr = larger_buffers[chunk_idx];

            LOG_DEBUG("  Chunk " << chunk_idx << ": exchanging " << this_chunk_elements
                                 << " elements (smaller offset: " << chunk_offset_elements << ")");

            // Create a temporary PCIeBAR exchange between these specific buffers
            // We need to:
            // 1. Read larger domain's chunk
            // 2. Add to smaller domain's chunk
            // 3. Write result back to both
            //
            // The bridge_backend_ is configured for cuda:0 ↔ rocm:0, but we may need
            // to exchange with rocm:1 as well. For now, use the staging approach:

            // Approach: Use bridge_backend_ for the exchange, but we need to handle
            // the fact that larger_buffers[chunk_idx] might not be on the bridge device.
            //
            // Actually, for the larger domain, after reduce-scatter each device has its
            // chunk. We need to exchange smaller↔larger[chunk_idx].
            //
            // If larger is ROCm: rocm_buffers[chunk_idx] is on rocm:chunk_idx
            // If smaller is CUDA: cuda_buffers[0] is on cuda:0
            //
            // The bridge is cuda:0 ↔ rocm:0. For chunk 0, this works directly.
            // For chunk 1 (rocm:1), we need to go through rocm:0 as a relay OR
            // use direct PCIe BAR access to rocm:1.
            //
            // For V1, let's use the bridge (rocm:0) as a staging point for non-bridge devices.

            DeviceId larger_device = larger_devices[chunk_idx];
            DeviceId smaller_device = smaller_devices[0];
            DeviceId smaller_bridge = plan.cuda_is_larger ? rocm_bridge_ : cuda_bridge_;
            DeviceId larger_bridge = plan.cuda_is_larger ? cuda_bridge_ : rocm_bridge_;

            // If this chunk is on the bridge device of the larger domain, direct exchange
            if (larger_device == larger_bridge)
            {
                // Direct bridge exchange
                LOG_DEBUG("    Direct bridge exchange (larger device is bridge)");

                if (!bridge_backend_->allreduce(
                        plan.cuda_is_larger ? larger_chunk_ptr : smaller_chunk_ptr,
                        this_chunk_elements, dtype, op))
                {
                    last_error_ = "Adaptive Phase 2: Bridge allreduce failed for chunk " +
                                  std::to_string(chunk_idx) + ": " + bridge_backend_->lastError();
                    LOG_ERROR("HeterogeneousBackend: " << last_error_);
                    return false;
                }
            }
            else
            {
                // Non-bridge device in larger domain. Need to relay through bridge.
                // This is more complex and requires staging.
                //
                // For chunk on larger[i] where i != 0:
                // 1. Copy larger[i]'s chunk to larger_bridge (e.g., NCCL send/recv or memcpy)
                // 2. Bridge exchange with smaller
                // 3. Copy result from larger_bridge back to larger[i]
                //
                // Actually, since we've already done reduce-scatter, larger[i] has the reduced
                // chunk. We need to:
                // - Send smaller's chunk[i] to larger[i] (via bridge staging)
                // - Receive larger[i]'s chunk to smaller (via bridge staging)
                // - Both sides add

                LOG_DEBUG("    Relayed exchange via bridge (larger device " << chunk_idx << " != bridge)");

                // Staging approach using the bridge device's buffer as temporary
                // We'll need temp space on the bridge devices.

                // For now, use a simpler approach: do sequential copies
                // 1. larger[i] → bridge (via intra-domain copy, e.g., NCCL sendrecv or cudaMemcpyPeer)
                // 2. bridge ↔ smaller (PCIeBAR allreduce)
                // 3. bridge → larger[i] (via intra-domain copy)

                // Get bridge buffer (larger[0] after reduce-scatter contains chunk 0, not what we want)
                // We need temp staging buffers. For V1, allocate temp or reuse existing.

                // SIMPLIFICATION for V1: Skip the relay complexity.
                // Fall back to standard 3-phase for chunks not on bridge device.
                // This means the adaptive pattern only helps if larger_domain_count == 1 or
                // we have direct PCIe access to all devices.

                // Actually, let's implement a proper relay using NCCL/RCCL send/recv.
                // But this requires significant additional complexity.

                // COMPROMISE for V1:
                // For non-bridge chunks, copy chunk to host, do host-mediated exchange.
                // This is slower but correct.

                LOG_WARN("Adaptive Phase 2: Non-bridge chunk " << chunk_idx
                                                               << " requires staging (not yet optimized)");

                // Allocate host staging buffers
                size_t chunk_bytes = this_chunk_elements * element_size;
                std::vector<char> larger_staging(chunk_bytes);
                std::vector<char> smaller_staging(chunk_bytes);

                // Copy larger[chunk_idx] to host
                if (plan.cuda_is_larger)
                {
                    // CUDA device - use CUDABackend or cudaMemcpy
                    // For now, assume we can use the backend's deviceToHost if available
                    // This is a limitation of V1 - we'd need direct CUDA/HIP calls here

                    // WORKAROUND: Use bridge_backend's internal copy capabilities
                    // Actually, PCIeBARBackend doesn't have this. We need raw CUDA/HIP.

                    // For V1, let's detect this case and fall back
                    LOG_WARN("Adaptive pattern: Non-bridge CUDA chunks require cudaMemcpy (falling back)");
                    return executeStandard3PhaseAllreduce(cuda_buffers, rocm_buffers, count, dtype, op);
                }
                else
                {
                    // ROCm device - similar issue
                    LOG_WARN("Adaptive pattern: Non-bridge ROCm chunks require hipMemcpy (falling back)");
                    return executeStandard3PhaseAllreduce(cuda_buffers, rocm_buffers, count, dtype, op);
                }
            }
        }

        LOG_DEBUG("HeterogeneousBackend Adaptive Phase2: Complete");

        // ═══════════════════════════════════════════════════════════════════
        // Phase 3: AllGather in larger domain
        // ═══════════════════════════════════════════════════════════════════
        // The larger domain devices each have one globally-reduced chunk.
        // AllGather reassembles the full tensor on all larger domain devices.
        // The smaller domain device already has the full tensor (from phase 2 exchanges).

        bool allgather_success = true;
        std::string allgather_error;

        if (plan.larger_domain_count > 1 && larger_backend)
        {
            LOG_DEBUG("HeterogeneousBackend Adaptive Phase3: "
                      << (plan.cuda_is_larger ? "NCCL" : "RCCL")
                      << " allgather");

            std::vector<const void *> send_bufs(larger_buffers.begin(), larger_buffers.end());
            allgather_success = larger_backend->allgatherMulti(
                send_bufs, const_cast<std::vector<void *> &>(larger_buffers),
                plan.chunk_elements, dtype);

            if (!allgather_success)
            {
                allgather_error = larger_backend->lastError();
            }
        }

        if (!allgather_success)
        {
            last_error_ = "Adaptive Phase 3: Larger domain allgather failed: " + allgather_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Adaptive Phase3: Complete");
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Partial Reduce-Scatter Pattern (for singleton configurations)
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // For singleton configurations (1 CUDA + N ROCm or N CUDA + 1 ROCm), the bridge
    // is a serial bottleneck. This pattern optimizes memory pressure in the larger
    // domain by using reduce-scatter, then staging chunks through the bridge.
    //
    // Example for 1 CUDA + 4 ROCm:
    //   Phase 1: RCCL reduce-scatter → ROCm[i] has chunk[i] (1/4 memory each)
    //   Phase 2: Chunked bridge exchange with staging:
    //            for i in 0..3:
    //              if i != 0: ROCm[i] → ROCm[0] via RCCL p2p
    //              ROCm[0] ↔ CUDA[0] via PCIeBAR (chunk[i])
    //              if i != 0: ROCm[0] → ROCm[i] via RCCL p2p
    //   Phase 3: RCCL allgather to reconstruct full tensor
    //

    bool HeterogeneousBackend::executePartialReduceScatter(
        std::vector<void *> &buffers,
        size_t count,
        CollectiveDataType dtype)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        // Determine which domain is singleton (1 device) and which is larger
        bool cuda_is_singleton = (cuda_devices_.size() == 1 && rocm_devices_.size() > 1);
        bool rocm_is_singleton = (rocm_devices_.size() == 1 && cuda_devices_.size() > 1);

        if (!cuda_is_singleton && !rocm_is_singleton)
        {
            // Not a singleton config - fall back to standard
            LOG_WARN("HeterogeneousBackend: executePartialReduceScatter called on non-singleton config");

            // Split buffers and use standard pattern
            std::vector<void *> cuda_buffers;
            std::vector<void *> rocm_buffers;
            for (size_t i = 0; i < device_group_.devices.size(); ++i)
            {
                if (device_group_.devices[i].is_cuda())
                    cuda_buffers.push_back(buffers[i]);
                else if (device_group_.devices[i].is_rocm())
                    rocm_buffers.push_back(buffers[i]);
            }
            return executeStandard3PhaseAllreduce(cuda_buffers, rocm_buffers, count, dtype, CollectiveOp::ALLREDUCE_SUM);
        }

        // Get element size for pointer arithmetic
        size_t element_size = collectiveDataTypeSize(dtype);

        // Identify singleton and larger domain
        size_t larger_count = std::max(cuda_devices_.size(), rocm_devices_.size());
        size_t chunk_size = count / larger_count;

        // Handle rounding - last chunk may be larger
        size_t last_chunk_extra = count % larger_count;

        LOG_DEBUG("HeterogeneousBackend: Executing PARTIAL reduce-scatter for singleton config");
        LOG_DEBUG("  Config: " << (cuda_is_singleton ? "1 CUDA + " : "")
                               << larger_count
                               << (rocm_is_singleton ? " CUDA + 1" : "") << " ROCm");
        LOG_DEBUG("  Total elements: " << count << ", chunk_size: " << chunk_size
                                       << ", last_chunk_extra: " << last_chunk_extra);

        // Split buffers by device type (maintaining order)
        std::vector<void *> cuda_buffers;
        std::vector<void *> rocm_buffers;
        void *cuda_bridge_buf = nullptr;
        void *rocm_bridge_buf = nullptr;

        for (size_t i = 0; i < device_group_.devices.size(); ++i)
        {
            const DeviceId &device = device_group_.devices[i];
            void *buf = buffers[i];

            if (device.is_cuda())
            {
                cuda_buffers.push_back(buf);
                if (device == cuda_bridge_)
                    cuda_bridge_buf = buf;
            }
            else if (device.is_rocm())
            {
                rocm_buffers.push_back(buf);
                if (device == rocm_bridge_)
                    rocm_bridge_buf = buf;
            }
        }

        // References for clarity
        const auto &larger_buffers = cuda_is_singleton ? rocm_buffers : cuda_buffers;
        const auto &singleton_buffers = cuda_is_singleton ? cuda_buffers : rocm_buffers;
        void *singleton_buffer = singleton_buffers[0];
        void *bridge_buffer = cuda_is_singleton ? rocm_bridge_buf : cuda_bridge_buf;

        // Get the appropriate backend for the larger domain
        ICollectiveBackend *larger_backend = cuda_is_singleton
                                                 ? static_cast<ICollectiveBackend *>(rccl_backend_.get())
                                                 : static_cast<ICollectiveBackend *>(nccl_backend_.get());

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 1: Reduce-scatter in larger domain
        // ═══════════════════════════════════════════════════════════════════════
        // Each device in the larger domain gets 1/N of the reduced data.

        LOG_DEBUG("HeterogeneousBackend Partial RS Phase1: Reduce-scatter in "
                  << (cuda_is_singleton ? "ROCm" : "CUDA") << " domain ("
                  << larger_count << " devices)");

        bool rs_success = true;
        std::string rs_error;

        if (larger_backend && larger_count > 1)
        {
            std::vector<const void *> send_bufs(larger_buffers.begin(), larger_buffers.end());
            rs_success = larger_backend->reduceScatterMulti(
                send_bufs, const_cast<std::vector<void *> &>(larger_buffers),
                chunk_size, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (!rs_success)
            {
                rs_error = larger_backend->lastError();
            }
        }

        if (!rs_success)
        {
            last_error_ = "Partial RS Phase 1: Reduce-scatter failed: " + rs_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Partial RS Phase1: Complete");

        // Synchronize before Phase 2 to ensure reduce-scatter results are available
        if (larger_backend)
        {
            larger_backend->synchronize();
            LOG_DEBUG("HeterogeneousBackend: Synchronized larger domain after Phase 1");
        }

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 2: Chunked bridge exchange with staging
        // ═══════════════════════════════════════════════════════════════════════
        // For each chunk:
        //   1. Stage chunk from larger[i] to bridge device (if i != bridge_idx)
        //   2. Bridge exchange: singleton ↔ bridge
        //   3. Stage result back to larger[i] (if i != bridge_idx)

        LOG_DEBUG("HeterogeneousBackend Partial RS Phase2: Chunked bridge exchange");

        if (!stageChunksThroughBridge(
                singleton_buffer,
                bridge_buffer,
                larger_buffers,
                chunk_size,
                dtype,
                cuda_is_singleton,
                CollectiveOp::ALLREDUCE_SUM))
        {
            // Error already logged in stageChunksThroughBridge
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Partial RS Phase2: Complete");

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 3: Allgather in larger domain
        // ═══════════════════════════════════════════════════════════════════════
        // Reconstruct the full tensor on all devices in the larger domain.
        // The singleton device already has the full tensor.

        LOG_DEBUG("HeterogeneousBackend Partial RS Phase3: Allgather in "
                  << (cuda_is_singleton ? "ROCm" : "CUDA") << " domain");

        bool ag_success = true;
        std::string ag_error;

        if (larger_backend && larger_count > 1)
        {
            std::vector<const void *> send_bufs(larger_buffers.begin(), larger_buffers.end());
            ag_success = larger_backend->allgatherMulti(
                send_bufs, const_cast<std::vector<void *> &>(larger_buffers),
                chunk_size, dtype);

            if (!ag_success)
            {
                ag_error = larger_backend->lastError();
            }
        }

        if (!ag_success)
        {
            last_error_ = "Partial RS Phase 3: Allgather failed: " + ag_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Partial RS Phase3: Complete");
        LOG_DEBUG("HeterogeneousBackend: Partial reduce-scatter allreduce complete");
        return true;
    }

    bool HeterogeneousBackend::stageChunksThroughBridge(
        void *singleton_buffer,
        void *bridge_buffer,
        const std::vector<void *> &chunk_buffers,
        size_t chunk_size,
        CollectiveDataType dtype,
        bool singleton_is_cuda,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        size_t element_size = collectiveDataTypeSize(dtype);
        size_t chunk_bytes = chunk_size * element_size;
        size_t num_chunks = chunk_buffers.size();

        // Get the backend for the larger domain (for p2p operations)
        ICollectiveBackend *larger_backend = singleton_is_cuda
                                                 ? static_cast<ICollectiveBackend *>(rccl_backend_.get())
                                                 : static_cast<ICollectiveBackend *>(nccl_backend_.get());

        // Determine bridge device index in the larger domain (always 0, lowest ordinal)
        constexpr size_t bridge_idx = 0;

        LOG_DEBUG("HeterogeneousBackend stageChunks: Processing " << num_chunks << " chunks, "
                                                                  << chunk_size << " elements each");

        // For each chunk, perform the staging and bridge exchange
        for (size_t i = 0; i < num_chunks; ++i)
        {
            size_t singleton_offset = i * chunk_size;
            void *singleton_chunk = static_cast<char *>(singleton_buffer) + singleton_offset * element_size;

            // After reduce-scatter, each device in the larger domain has exactly one chunk
            // chunk_buffers[i] points to the start of that device's buffer, which now contains chunk[i]
            void *chunk_ptr = chunk_buffers[i];

            LOG_DEBUG("  Chunk " << i << ": singleton_offset=" << singleton_offset
                                 << ", chunk_ptr=" << chunk_ptr);

            // ─────────────────────────────────────────────────────────────────
            // Step 1: Stage chunk to bridge device if not already there
            // ─────────────────────────────────────────────────────────────────
            if (i != bridge_idx)
            {
                // Use sendrecvMulti: device[i] sends chunk to device[bridge_idx]
                // This properly coordinates NCCL/RCCL by issuing both endpoints in one group
                LOG_DEBUG("    Step 1: Stage chunk " << i << " to bridge device via sendrecvMulti");

                if (larger_backend)
                {
                    // Transfer chunk from device i to bridge device (index 0)
                    // sendrecvMulti handles the NCCL/RCCL group coordination
                    bool ok = larger_backend->sendrecvMulti(
                        chunk_ptr,     // src: chunk on device i
                        bridge_buffer, // dst: staging buffer on bridge device
                        chunk_size,
                        dtype,
                        static_cast<int>(i),           // src_gpu: device i
                        static_cast<int>(bridge_idx)); // dst_gpu: bridge device (0)

                    if (!ok)
                    {
                        last_error_ = "stageChunks: sendrecvMulti failed for chunk " + std::to_string(i) +
                                      " -> bridge: " + larger_backend->lastError();
                        LOG_ERROR("HeterogeneousBackend: " << last_error_);
                        return false;
                    }
                }
                else
                {
                    // No larger backend (shouldn't happen in singleton config with >1 device)
                    last_error_ = "stageChunks: Larger domain backend not available";
                    LOG_ERROR("HeterogeneousBackend: " << last_error_);
                    return false;
                }
            }
            else
            {
                // Chunk is already on bridge device - use it directly
                // bridge_buffer and chunk_buffers[0] should be the same
                LOG_DEBUG("    Step 1: Chunk " << i << " already on bridge device");
            }

            // ─────────────────────────────────────────────────────────────────
            // Step 2: Bridge exchange (singleton ↔ bridge)
            // ─────────────────────────────────────────────────────────────────
            // Exchange this chunk between singleton device and bridge device
            // Both sides accumulate (add their local data to received data)

            LOG_DEBUG("    Step 2: Bridge exchange for chunk " << i);

            // Determine which buffer to use for the bridge exchange
            // If i == bridge_idx, use chunk_buffers[i] directly
            // Otherwise, use bridge_buffer (which now has chunk[i] staged to it)
            void *larger_bridge_chunk = (i == bridge_idx) ? chunk_ptr : bridge_buffer;

            // For the singleton side, we exchange chunk[i] portion of its full buffer
            void *cuda_buf = singleton_is_cuda ? singleton_chunk : larger_bridge_chunk;
            void *rocm_buf = singleton_is_cuda ? larger_bridge_chunk : singleton_chunk;

            if (!executePhase2_BridgeExchange(cuda_buf, rocm_buf, chunk_size, dtype, op))
            {
                last_error_ = "stageChunks: Bridge exchange failed for chunk " + std::to_string(i) +
                              ": " + last_error_;
                LOG_ERROR("HeterogeneousBackend: " << last_error_);
                return false;
            }

            // ─────────────────────────────────────────────────────────────────
            // Step 3: Stage result back to original device if needed
            // ─────────────────────────────────────────────────────────────────
            if (i != bridge_idx)
            {
                LOG_DEBUG("    Step 3: Stage result back to device " << i << " via sendrecvMulti");

                if (larger_backend)
                {
                    // Transfer result from bridge device back to device i
                    // sendrecvMulti handles the NCCL/RCCL group coordination
                    bool ok = larger_backend->sendrecvMulti(
                        bridge_buffer, // src: result on bridge device
                        chunk_ptr,     // dst: original chunk location on device i
                        chunk_size,
                        dtype,
                        static_cast<int>(bridge_idx), // src_gpu: bridge device (0)
                        static_cast<int>(i));         // dst_gpu: device i

                    if (!ok)
                    {
                        last_error_ = "stageChunks: sendrecvMulti failed for bridge -> chunk " +
                                      std::to_string(i) + ": " + larger_backend->lastError();
                        LOG_ERROR("HeterogeneousBackend: " << last_error_);
                        return false;
                    }
                }
            }
            else
            {
                LOG_DEBUG("    Step 3: Result already on correct device " << i);
            }
        }

        LOG_DEBUG("HeterogeneousBackend stageChunks: All " << num_chunks << " chunks processed");
        return true;
    }

    bool HeterogeneousBackend::stageChunksThroughBridgePipelined(
        void *singleton_buffer,
        void *bridge_buffer_0,
        void *bridge_buffer_1,
        const std::vector<void *> &chunk_buffers,
        size_t chunk_size,
        CollectiveDataType dtype,
        bool singleton_is_cuda,
        CollectiveOp op,
        size_t pipeline_depth)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        size_t element_size = collectiveDataTypeSize(dtype);
        size_t chunk_bytes = chunk_size * element_size;
        size_t num_chunks = chunk_buffers.size();

        // Validate pipeline depth (minimum 2 for double buffering)
        if (pipeline_depth < 2)
        {
            pipeline_depth = 2;
        }

        // Get the backend for the larger domain (for p2p operations)
        ICollectiveBackend *larger_backend = singleton_is_cuda
                                                 ? static_cast<ICollectiveBackend *>(rccl_backend_.get())
                                                 : static_cast<ICollectiveBackend *>(nccl_backend_.get());

        // Determine bridge device index in the larger domain (always 0, lowest ordinal)
        constexpr size_t bridge_idx = 0;

        LOG_DEBUG("HeterogeneousBackend stageChunksPipelined: Processing " << num_chunks
                                                                           << " chunks with pipeline depth " << pipeline_depth
                                                                           << ", " << chunk_size << " elements each");

        // ═══════════════════════════════════════════════════════════════════════
        // Pipelined Processing with Double Buffering
        // ═══════════════════════════════════════════════════════════════════════
        //
        // For N chunks, we overlap operations using two staging buffers:
        //
        // Timeline (simplified for pipeline_depth=2):
        // ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
        // │   Iter 0    │   Iter 1    │   Iter 2    │   Iter 3    │   Iter 4    │
        // ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
        // │ StageIn[0]  │ StageIn[1]  │ StageIn[2]  │ ─           │ ─           │
        // │ ─           │ Bridge[0]   │ Bridge[1]   │ Bridge[2]   │ ─           │
        // │ ─           │ ─           │ StageOut[0] │ StageOut[1] │ StageOut[2] │
        // └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
        //
        // Buffer assignment (alternating):
        //   Chunk 0, 2, 4, ... → bridge_buffer_0
        //   Chunk 1, 3, 5, ... → bridge_buffer_1
        // ═══════════════════════════════════════════════════════════════════════

        // For simplicity in V1, we implement a quasi-pipelined approach:
        // - Stage chunk i into buffer[i%2]
        // - Wait for previous bridge exchange (if any) to complete
        // - Start bridge exchange for chunk i
        // - Stage out result for chunk i-1 (if applicable)
        //
        // True async overlap would require explicit stream/event management.
        // This implementation provides the structure for future optimization.

        // Track which buffer is in use for bridge exchange
        void *bridge_buffers[2] = {bridge_buffer_0, bridge_buffer_1};

        // Track completed bridge exchanges for stage-out
        std::vector<bool> bridge_complete(num_chunks, false);
        std::vector<void *> staged_results(num_chunks, nullptr);

        // Process chunks in pipelined fashion
        for (size_t iter = 0; iter < num_chunks + pipeline_depth - 1; ++iter)
        {
            // ─────────────────────────────────────────────────────────────────
            // Stage-In: Stage chunk[iter] to bridge (if iter < num_chunks)
            // ─────────────────────────────────────────────────────────────────
            if (iter < num_chunks)
            {
                size_t chunk_idx = iter;
                void *staging_buf = bridge_buffers[chunk_idx % 2];
                void *chunk_ptr = chunk_buffers[chunk_idx];

                LOG_DEBUG("  Pipeline iter " << iter << ": Stage-in chunk " << chunk_idx);

                if (chunk_idx != bridge_idx)
                {
                    // Stage chunk from device[chunk_idx] to bridge using sendrecvMulti
                    if (larger_backend)
                    {
                        // sendrecvMulti coordinates NCCL/RCCL group properly
                        bool ok = larger_backend->sendrecvMulti(
                            chunk_ptr,   // src: chunk on device
                            staging_buf, // dst: staging buffer on bridge
                            chunk_size,
                            dtype,
                            static_cast<int>(chunk_idx),   // src_gpu
                            static_cast<int>(bridge_idx)); // dst_gpu

                        if (!ok)
                        {
                            last_error_ = "stageChunksPipelined: sendrecvMulti stage-in failed for chunk " +
                                          std::to_string(chunk_idx) + ": " + larger_backend->lastError();
                            LOG_ERROR("HeterogeneousBackend: " << last_error_);
                            return false;
                        }
                    }
                    staged_results[chunk_idx] = staging_buf;
                }
                else
                {
                    // Chunk is already on bridge device
                    staged_results[chunk_idx] = chunk_ptr;
                }
            }

            // ─────────────────────────────────────────────────────────────────
            // Bridge Exchange: Process chunk[iter-1] (if iter > 0 && iter-1 < num_chunks)
            // ─────────────────────────────────────────────────────────────────
            if (iter > 0 && iter - 1 < num_chunks)
            {
                size_t chunk_idx = iter - 1;
                size_t singleton_offset = chunk_idx * chunk_size;
                void *singleton_chunk = static_cast<char *>(singleton_buffer) +
                                        singleton_offset * element_size;

                LOG_DEBUG("  Pipeline iter " << iter << ": Bridge exchange for chunk " << chunk_idx);

                // Get the staged buffer (either staging buffer or direct chunk if bridge device)
                void *larger_bridge_chunk = staged_results[chunk_idx];

                // For the singleton side, we exchange chunk[i] portion of its full buffer
                void *cuda_buf = singleton_is_cuda ? singleton_chunk : larger_bridge_chunk;
                void *rocm_buf = singleton_is_cuda ? larger_bridge_chunk : singleton_chunk;

                // TODO: Use async bridge operations for true overlap
                if (!executePhase2_BridgeExchange(cuda_buf, rocm_buf, chunk_size, dtype, op))
                {
                    last_error_ = "stageChunksPipelined: Bridge exchange failed for chunk " +
                                  std::to_string(chunk_idx) + ": " + last_error_;
                    LOG_ERROR("HeterogeneousBackend: " << last_error_);
                    return false;
                }

                bridge_complete[chunk_idx] = true;
            }

            // ─────────────────────────────────────────────────────────────────
            // Stage-Out: Write result back for chunk[iter-2] (if applicable)
            // ─────────────────────────────────────────────────────────────────
            if (iter >= 2 && iter - 2 < num_chunks)
            {
                size_t chunk_idx = iter - 2;

                if (!bridge_complete[chunk_idx])
                {
                    // Bridge not yet complete - skip (shouldn't happen in serial execution)
                    continue;
                }

                LOG_DEBUG("  Pipeline iter " << iter << ": Stage-out result for chunk " << chunk_idx);

                if (chunk_idx != bridge_idx)
                {
                    void *result_buf = staged_results[chunk_idx];
                    void *chunk_ptr = chunk_buffers[chunk_idx];

                    if (larger_backend)
                    {
                        // sendrecvMulti coordinates NCCL/RCCL group properly
                        bool ok = larger_backend->sendrecvMulti(
                            result_buf, // src: result on bridge
                            chunk_ptr,  // dst: original chunk location
                            chunk_size,
                            dtype,
                            static_cast<int>(bridge_idx), // src_gpu
                            static_cast<int>(chunk_idx)); // dst_gpu

                        if (!ok)
                        {
                            last_error_ = "stageChunksPipelined: sendrecvMulti stage-out failed for chunk " +
                                          std::to_string(chunk_idx) + ": " + larger_backend->lastError();
                            LOG_ERROR("HeterogeneousBackend: " << last_error_);
                            return false;
                        }
                    }
                }
                // else: chunk is on bridge device, result already in place
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Drain: Process any remaining chunks
        // ─────────────────────────────────────────────────────────────────────
        // Handle the last chunk's bridge exchange and stage-out
        if (num_chunks > 0)
        {
            size_t last_chunk_idx = num_chunks - 1;

            // Complete bridge exchange for last chunk if not done
            if (!bridge_complete[last_chunk_idx])
            {
                size_t singleton_offset = last_chunk_idx * chunk_size;
                void *singleton_chunk = static_cast<char *>(singleton_buffer) +
                                        singleton_offset * element_size;
                void *larger_bridge_chunk = staged_results[last_chunk_idx];
                void *cuda_buf = singleton_is_cuda ? singleton_chunk : larger_bridge_chunk;
                void *rocm_buf = singleton_is_cuda ? larger_bridge_chunk : singleton_chunk;

                if (!executePhase2_BridgeExchange(cuda_buf, rocm_buf, chunk_size, dtype, op))
                {
                    last_error_ = "stageChunksPipelined: Bridge exchange failed for last chunk: " +
                                  last_error_;
                    LOG_ERROR("HeterogeneousBackend: " << last_error_);
                    return false;
                }
                bridge_complete[last_chunk_idx] = true;
            }

            // Stage out any remaining results
            for (size_t i = 0; i < num_chunks; ++i)
            {
                if (i != bridge_idx && bridge_complete[i])
                {
                    // Check if already staged out (from main loop)
                    // We need to stage out chunks that weren't covered in the main loop
                    size_t main_loop_iters = num_chunks + pipeline_depth - 1;
                    size_t stageout_start_iter = 2; // Stage-out starts at iter 2
                    size_t stageout_end_iter = main_loop_iters;

                    // Calculate which chunks were staged out in main loop
                    // Stage-out at iter k processes chunk k-2
                    // So main loop stages out chunks 0 to (main_loop_iters - 2 - 1)
                    size_t max_stageout_chunk_in_loop = (main_loop_iters >= 2) ? main_loop_iters - 2 - 1 : 0;

                    if (i > max_stageout_chunk_in_loop || main_loop_iters < 2)
                    {
                        void *result_buf = staged_results[i];
                        void *chunk_ptr = chunk_buffers[i];

                        if (larger_backend)
                        {
                            // sendrecvMulti coordinates NCCL/RCCL group properly
                            bool ok = larger_backend->sendrecvMulti(
                                result_buf, // src: result on bridge
                                chunk_ptr,  // dst: original chunk location
                                chunk_size,
                                dtype,
                                static_cast<int>(bridge_idx), // src_gpu
                                static_cast<int>(i));         // dst_gpu

                            if (!ok)
                            {
                                last_error_ = "stageChunksPipelined: Drain sendrecvMulti failed for chunk " +
                                              std::to_string(i) + ": " + larger_backend->lastError();
                                LOG_ERROR("HeterogeneousBackend: " << last_error_);
                                return false;
                            }
                        }
                    }
                }
            }
        }

        LOG_DEBUG("HeterogeneousBackend stageChunksPipelined: All " << num_chunks
                                                                    << " chunks processed");
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 2→3 Chunk-Based Pipelining
    // ═══════════════════════════════════════════════════════════════════════════

    HeterogeneousBackend::PipelinePlan HeterogeneousBackend::planPipelining(
        size_t count, size_t element_size) const
    {
        PipelinePlan plan;
        plan.total_elements = count;

        size_t total_bytes = count * element_size;

        // Check if tensor is large enough for pipelining
        if (total_bytes < PIPELINE_MIN_TENSOR_SIZE)
        {
            // Small tensor: single chunk, no pipelining benefit
            plan.will_use_pipelining = false;
            plan.num_chunks = 1;
            plan.chunk_elements = count;
            plan.last_chunk_elements = count;
            return plan;
        }

        // Calculate chunk parameters
        size_t chunk_bytes = PIPELINE_CHUNK_SIZE;
        size_t elements_per_chunk = chunk_bytes / element_size;

        // Ensure at least 1 element per chunk
        if (elements_per_chunk == 0)
        {
            elements_per_chunk = 1;
        }

        // Calculate number of chunks
        size_t num_chunks = (count + elements_per_chunk - 1) / elements_per_chunk;

        // Calculate last chunk size (may be smaller due to rounding)
        size_t full_chunks = count / elements_per_chunk;
        size_t remainder = count % elements_per_chunk;
        size_t last_chunk_elements = (remainder > 0) ? remainder : elements_per_chunk;

        plan.will_use_pipelining = (num_chunks > 1);
        plan.num_chunks = num_chunks;
        plan.chunk_elements = elements_per_chunk;
        plan.last_chunk_elements = last_chunk_elements;

        return plan;
    }

    bool HeterogeneousBackend::executeReduceScatterPatternPipelined(
        const std::vector<void *> &cuda_buffers,
        const std::vector<void *> &rocm_buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        // ═══════════════════════════════════════════════════════════════════
        // CRITICAL: Check for asymmetric device counts
        // ═══════════════════════════════════════════════════════════════════
        // Reduce-scatter pattern requires symmetric device counts. With asymmetric
        // counts (e.g., 1 CUDA + 2 ROCm), the chunking doesn't align and data
        // ends up unreduced. Fall back to standard 3-phase pattern.
        if (cuda_devices_.size() != rocm_devices_.size())
        {
            LOG_DEBUG("HeterogeneousBackend: Asymmetric device counts (CUDA="
                      << cuda_devices_.size() << ", ROCm=" << rocm_devices_.size()
                      << "), using standard 3-phase pattern instead of pipelined reduce-scatter");
            return executeStandard3PhaseAllreduce(cuda_buffers, rocm_buffers, count, dtype, op);
        }

        size_t element_size = collectiveDataTypeSize(dtype);
        auto plan = planPipelining(count, element_size);

        LOG_DEBUG("HeterogeneousBackend: Pipelined reduce-scatter for " << count
                                                                        << " elements (pipelining=" << (plan.will_use_pipelining ? "ON" : "OFF")
                                                                        << ", chunks=" << plan.num_chunks << ")");

        // If pipelining is not beneficial, fall back to standard pattern
        if (!plan.will_use_pipelining)
        {
            LOG_DEBUG("HeterogeneousBackend: Tensor too small for pipelining, using standard pattern");
            return executeReduceScatterPattern(cuda_buffers, rocm_buffers, count, dtype, op);
        }

        // Compute per-domain chunk sizes
        size_t cuda_chunk_count = count / cuda_devices_.size();
        size_t rocm_chunk_count = count / rocm_devices_.size();

        // ═══════════════════════════════════════════════════════════════════
        // Phase 1: Reduce-Scatter in each domain (unchanged from standard)
        // ═══════════════════════════════════════════════════════════════════

        bool nccl_success = true;
        bool rccl_success = true;
        std::string nccl_error;
        std::string rccl_error;

        std::thread nccl_thread;
        std::thread rccl_thread;

        // NCCL reduce-scatter (if >1 CUDA device)
        if (cuda_devices_.size() > 1 && nccl_backend_)
        {
            LOG_DEBUG("HeterogeneousBackend Pipelined Phase1: NCCL reduce-scatter with "
                      << cuda_devices_.size() << " devices, chunk=" << cuda_chunk_count);

            nccl_thread = std::thread([&]()
                                      {
                std::vector<const void*> send_bufs(cuda_buffers.begin(), cuda_buffers.end());
                nccl_success = nccl_backend_->reduceScatterMulti(
                    send_bufs, cuda_buffers, cuda_chunk_count, dtype, op);
                if (!nccl_success) {
                    nccl_error = nccl_backend_->lastError();
                } });
        }

        // RCCL reduce-scatter (if >1 ROCm device)
        if (rocm_devices_.size() > 1 && rccl_backend_)
        {
            LOG_DEBUG("HeterogeneousBackend Pipelined Phase1: RCCL reduce-scatter with "
                      << rocm_devices_.size() << " devices, chunk=" << rocm_chunk_count);

            rccl_thread = std::thread([&]()
                                      {
                std::vector<const void*> send_bufs(rocm_buffers.begin(), rocm_buffers.end());
                rccl_success = rccl_backend_->reduceScatterMulti(
                    send_bufs, rocm_buffers, rocm_chunk_count, dtype, op);
                if (!rccl_success) {
                    rccl_error = rccl_backend_->lastError();
                } });
        }

        if (nccl_thread.joinable())
            nccl_thread.join();
        if (rccl_thread.joinable())
            rccl_thread.join();

        if (!nccl_success)
        {
            last_error_ = "Pipelined Phase 1: NCCL reduce-scatter failed: " + nccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }
        if (!rccl_success)
        {
            last_error_ = "Pipelined Phase 1: RCCL reduce-scatter failed: " + rccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Pipelined Phase1: Reduce-scatter complete");

        // ═══════════════════════════════════════════════════════════════════
        // Phase 2+3 Pipelined: Chunked bridge exchange + allgather
        // ═══════════════════════════════════════════════════════════════════
        // V1 Implementation: Serial per-chunk execution
        // Future: True async overlap with CUDA/HIP streams

        void *cuda_bridge_buf = cuda_buffers[0];
        void *rocm_bridge_buf = rocm_buffers[0];

        // The bridge exchange count is the minimum chunk size after reduce-scatter
        size_t bridge_total_count = std::min(cuda_chunk_count, rocm_chunk_count);

        // Plan pipelining for bridge exchange
        auto bridge_plan = planPipelining(bridge_total_count, element_size);

        LOG_DEBUG("HeterogeneousBackend Pipelined Phase2+3: "
                  << bridge_plan.num_chunks << " chunks of "
                  << bridge_plan.chunk_elements << " elements each");

        // Process each chunk: bridge exchange then allgather
        for (size_t chunk_idx = 0; chunk_idx < bridge_plan.num_chunks; ++chunk_idx)
        {
            size_t chunk_offset_elements = chunk_idx * bridge_plan.chunk_elements;
            size_t this_chunk_elements = (chunk_idx == bridge_plan.num_chunks - 1)
                                             ? bridge_plan.last_chunk_elements
                                             : bridge_plan.chunk_elements;

            // Calculate byte offset for this chunk
            size_t byte_offset = chunk_offset_elements * element_size;

            // Get pointers to this chunk's data (both CUDA and ROCm)
            char *cuda_chunk = static_cast<char *>(cuda_bridge_buf) + byte_offset;
            char *rocm_chunk = static_cast<char *>(rocm_bridge_buf) + byte_offset;

            LOG_DEBUG("HeterogeneousBackend Pipelined: Chunk " << chunk_idx
                                                               << "/" << bridge_plan.num_chunks
                                                               << " offset=" << chunk_offset_elements
                                                               << " elements=" << this_chunk_elements);

            // ─────────────────────────────────────────────────────────────────
            // Phase 2 for this chunk: Bridge exchange using staging buffer pattern
            // ─────────────────────────────────────────────────────────────────
            // CRITICAL: Use executePhase2_BridgeExchange which knows about BOTH
            // buffers and uses the BAR staging pattern. The old call to
            // bridge_backend_->allreduce(cuda_chunk, ...) was broken because
            // PCIeBARBackend reads ROCm data from BAR offset 0, but hipMalloc
            // buffers are at different VRAM offsets.
            if (!executePhase2_BridgeExchange(cuda_chunk, rocm_chunk, this_chunk_elements, dtype, op))
            {
                last_error_ = "Pipelined Phase 2: Bridge exchange failed for chunk " +
                              std::to_string(chunk_idx) + ": " + last_error_;
                LOG_ERROR("HeterogeneousBackend: " << last_error_);
                return false;
            }

            // ─────────────────────────────────────────────────────────────────
            // Phase 3 for this chunk: Allgather (serial for V1)
            // ─────────────────────────────────────────────────────────────────
            // V1: Serial execution - allgather for chunk N completes before
            //     bridge exchange for chunk N+1.
            // V2 TODO: Use async allgather with CUDA/HIP streams for true overlap

            nccl_success = true;
            rccl_success = true;

            // NCCL allgather for this chunk (if >1 CUDA device)
            if (cuda_devices_.size() > 1 && nccl_backend_)
            {
                nccl_thread = std::thread([&, chunk_offset_elements, this_chunk_elements]()
                                          {
                    // Build chunk-specific buffers
                    std::vector<const void*> send_bufs;
                    std::vector<void*> recv_bufs;
                    for (size_t i = 0; i < cuda_buffers.size(); ++i) {
                        char* buf_base = static_cast<char*>(cuda_buffers[i]);
                        size_t buf_offset = chunk_offset_elements * element_size;
                        send_bufs.push_back(buf_base + buf_offset);
                        recv_bufs.push_back(buf_base + buf_offset);
                    }
                    nccl_success = nccl_backend_->allgatherMulti(
                        send_bufs, recv_bufs, this_chunk_elements, dtype);
                    if (!nccl_success) {
                        nccl_error = nccl_backend_->lastError();
                    } });
            }

            // RCCL allgather for this chunk (if >1 ROCm device)
            if (rocm_devices_.size() > 1 && rccl_backend_)
            {
                rccl_thread = std::thread([&, chunk_offset_elements, this_chunk_elements]()
                                          {
                    // Build chunk-specific buffers
                    std::vector<const void*> send_bufs;
                    std::vector<void*> recv_bufs;
                    for (size_t i = 0; i < rocm_buffers.size(); ++i) {
                        char* buf_base = static_cast<char*>(rocm_buffers[i]);
                        size_t buf_offset = chunk_offset_elements * element_size;
                        send_bufs.push_back(buf_base + buf_offset);
                        recv_bufs.push_back(buf_base + buf_offset);
                    }
                    rccl_success = rccl_backend_->allgatherMulti(
                        send_bufs, recv_bufs, this_chunk_elements, dtype);
                    if (!rccl_success) {
                        rccl_error = rccl_backend_->lastError();
                    } });
            }

            if (nccl_thread.joinable())
                nccl_thread.join();
            if (rccl_thread.joinable())
                rccl_thread.join();

            if (!nccl_success)
            {
                last_error_ = "Pipelined Phase 3: NCCL allgather failed for chunk " +
                              std::to_string(chunk_idx) + ": " + nccl_error;
                LOG_ERROR("HeterogeneousBackend: " << last_error_);
                return false;
            }
            if (!rccl_success)
            {
                last_error_ = "Pipelined Phase 3: RCCL allgather failed for chunk " +
                              std::to_string(chunk_idx) + ": " + rccl_error;
                LOG_ERROR("HeterogeneousBackend: " << last_error_);
                return false;
            }
        }

        LOG_DEBUG("HeterogeneousBackend Pipelined Phase2+3: Complete ("
                  << bridge_plan.num_chunks << " chunks processed)");
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 2: Cross-Domain Bridge Exchange
    // ═══════════════════════════════════════════════════════════════════════════

    HeterogeneousBackend::Phase2Plan HeterogeneousBackend::planPhase2() const
    {
        Phase2Plan plan;

        // Phase 2 always calls bridge allreduce if initialized
        // (The bridge_backend_ is always created for heterogeneous configs)
        if (initialized_ && bridge_backend_)
        {
            plan.will_call_bridge_allreduce = true;
            plan.cuda_bridge_device = cuda_bridge_;
            plan.rocm_bridge_device = rocm_bridge_;
        }
        else
        {
            // Not initialized - return defaults (cpu devices, no allreduce)
            plan.cuda_bridge_device = DeviceId::cpu();
            plan.rocm_bridge_device = DeviceId::cpu();
        }

        return plan;
    }

    bool HeterogeneousBackend::executePhase2_BridgeExchange(
        void *cuda_bridge_buffer,
        void *rocm_bridge_buffer,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        if (!bridge_backend_)
        {
            last_error_ = "Bridge backend not initialized";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        if (!cuda_bridge_buffer || !rocm_bridge_buffer)
        {
            last_error_ = "Phase 2: Null bridge buffer";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Phase2: Starting bridge exchange between "
                  << cuda_bridge_.toString() << " and " << rocm_bridge_.toString()
                  << " (count=" << count << ")");

        // ═══════════════════════════════════════════════════════════════════
        // CRITICAL: Staging Buffer Pattern for Cross-Vendor Allreduce
        // ═══════════════════════════════════════════════════════════════════
        // The ROCm buffer is allocated via hipMalloc(), which places it in GPU
        // VRAM at an arbitrary offset. The PCIeBAR backend maps only a portion
        // of the AMD VRAM (1GB at offset 0), so hipMalloc buffers are NOT
        // directly accessible via BAR.
        //
        // Solution: Use the BAR mmap as a staging area:
        // 1. hipMemcpy(D2D) ROCm buffer → BAR mmap (uses AMD DMA engine, ~4 GB/s)
        // 2. CUDA reduces: cuda_buffer + BAR_data → cuda_buffer
        // 3. cudaMemcpy(D2D) cuda_buffer → BAR mmap (CUDA writes to BAR)
        // 4. hipMemcpy(D2D) BAR mmap → ROCm buffer (AMD DMA reads from BAR)
        //
        // This is the known-good pattern from LocalTPContext::executePCIeBarAllreduce.
        // ═══════════════════════════════════════════════════════════════════

        size_t bytes = count * collectiveDataTypeSize(dtype);

        // Get the BAR staging pointer
        void *bar_staging_ptr = bridge_backend_->getBarHostPtr();
        if (!bar_staging_ptr)
        {
            last_error_ = "Phase 2: BAR staging pointer not available";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        // Check BAR has enough space
        if (bytes > bridge_backend_->getBarTotalMappedSize())
        {
            last_error_ = "Phase 2: Transfer size (" + std::to_string(bytes) +
                          ") exceeds BAR mapped size (" +
                          std::to_string(bridge_backend_->getBarTotalMappedSize()) + ")";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Phase2: Using BAR staging pattern "
                  << "(rocm_buf=" << rocm_bridge_buffer << ", bar=" << bar_staging_ptr
                  << ", cuda_buf=" << cuda_bridge_buffer << ", bytes=" << bytes << ")");

        // Get ROCmBackend for HIP memory operations (avoids HIP header conflicts)
        ROCmBackend *rocm_backend = dynamic_cast<ROCmBackend *>(getROCmBackend());
        if (!rocm_backend)
        {
            last_error_ = "Phase 2: ROCmBackend not available";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        int rocm_ordinal = rocm_bridge_.ordinal;

        // Step 1: Copy ROCm data to BAR staging via ROCmBackend::deviceToDevice
        // The BAR mmap is accessible to HIP as a "device" pointer for D2D copies
        rocm_backend->setDevice(rocm_ordinal);
        if (!rocm_backend->deviceToDevice(bar_staging_ptr, rocm_bridge_buffer, bytes, rocm_ordinal))
        {
            last_error_ = "Phase 2: ROCm → BAR copy failed";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }
        rocm_backend->synchronize(rocm_ordinal);

        LOG_DEBUG("HeterogeneousBackend Phase2: Step 1 complete - ROCm data staged to BAR");

        // Step 2: CUDA reduction - read from BAR, accumulate into CUDA buffer
        // cuda_buffer = cuda_buffer + bar_staging
        if (!bridge_backend_->reduceOnCUDA(cuda_bridge_buffer, cuda_bridge_buffer, bar_staging_ptr,
                                           count, dtype, op))
        {
            last_error_ = "Phase 2: CUDA reduction failed: " + bridge_backend_->lastError();
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Phase2: Step 2 complete - CUDA reduction done");

        // Step 3: Write CUDA result to BAR staging using DirectP2PEngine
        // CRITICAL: Use the DirectP2PEngine's transferViaPCIeBar() method instead of
        // cudaMemcpy because it properly uses the CUDA driver API (cuMemcpyDtoD) with
        // the registered IOMEMORY pointer, ensuring correct cache coherency for
        // PCIe BAR writes. Using cudaMemcpy with a host-registered IOMEMORY pointer
        // can leave tail data unflushed, causing partial transfer failures.
        auto *p2p_engine = bridge_backend_->getDirectP2PEngine();
        if (!p2p_engine || !p2p_engine->isPCIeBarActive())
        {
            last_error_ = "Phase 2: DirectP2P engine not available";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        auto transfer_result = p2p_engine->transferViaPCIeBar(
            cuda_bridge_buffer, 0, bytes, DirectP2PEngine::Direction::ToAMD);
        if (!transfer_result.success)
        {
            last_error_ = "Phase 2: CUDA → BAR transfer failed: " + transfer_result.error_message;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Phase2: Step 3 complete - Result written to BAR");

        // Step 4: Copy from BAR staging to ROCm buffer via ROCmBackend::deviceToDevice
        rocm_backend->setDevice(rocm_ordinal);
        if (!rocm_backend->deviceToDevice(rocm_bridge_buffer, bar_staging_ptr, bytes, rocm_ordinal))
        {
            last_error_ = "Phase 2: BAR → ROCm copy failed";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }
        rocm_backend->synchronize(rocm_ordinal);

        LOG_DEBUG("HeterogeneousBackend Phase2: Step 4 complete - ROCm buffer updated");

        bool success = true; // All steps completed

        if (!success)
        {
            last_error_ = "Phase 2: Bridge allreduce failed: " + bridge_backend_->lastError();
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Phase2: Bridge exchange complete");
        return true;
    }

    bool HeterogeneousBackend::executePhase1_IntraDomainReduce(
        const std::vector<void *> &cuda_buffers,
        const std::vector<void *> &rocm_buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        // Validate buffer counts match device counts
        if (cuda_buffers.size() != cuda_devices_.size())
        {
            last_error_ = "CUDA buffer count (" + std::to_string(cuda_buffers.size()) +
                          ") does not match CUDA device count (" +
                          std::to_string(cuda_devices_.size()) + ")";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        if (rocm_buffers.size() != rocm_devices_.size())
        {
            last_error_ = "ROCm buffer count (" + std::to_string(rocm_buffers.size()) +
                          ") does not match ROCm device count (" +
                          std::to_string(rocm_devices_.size()) + ")";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        Phase1Plan plan = planPhase1();
        bool nccl_success = true;
        bool rccl_success = true;
        std::string nccl_error;
        std::string rccl_error;

        // Execute NCCL and RCCL reductions in parallel using std::thread
        // They operate on independent hardware domains, so this is safe
        std::thread nccl_thread;
        std::thread rccl_thread;

        if (plan.will_call_nccl_reduce)
        {
            LOG_DEBUG("HeterogeneousBackend Phase1: Starting NCCL reduce with "
                      << cuda_buffers.size() << " CUDA buffers to root=" << plan.nccl_reduce_root);
            nccl_thread = std::thread([&]()
                                      {
                // Cast away const for the buffer vector (reduceMulti takes non-const)
                std::vector<void*> mutable_buffers(cuda_buffers.begin(), cuda_buffers.end());
                nccl_success = nccl_backend_->reduceMulti(
                    mutable_buffers, count, dtype, op, plan.nccl_reduce_root);
                if (!nccl_success) {
                    nccl_error = nccl_backend_->lastError();
                } });
        }

        if (plan.will_call_rccl_reduce)
        {
            LOG_DEBUG("HeterogeneousBackend Phase1: Starting RCCL reduce with "
                      << rocm_buffers.size() << " ROCm buffers to root=" << plan.rccl_reduce_root);
            rccl_thread = std::thread([&]()
                                      {
                // Cast away const for the buffer vector (reduceMulti takes non-const)
                std::vector<void*> mutable_buffers(rocm_buffers.begin(), rocm_buffers.end());
                rccl_success = rccl_backend_->reduceMulti(
                    mutable_buffers, count, dtype, op, plan.rccl_reduce_root);
                if (!rccl_success) {
                    rccl_error = rccl_backend_->lastError();
                } });
        }

        // Wait for both threads to complete
        if (nccl_thread.joinable())
        {
            nccl_thread.join();
        }
        if (rccl_thread.joinable())
        {
            rccl_thread.join();
        }

        // Check results
        if (!nccl_success)
        {
            last_error_ = "NCCL intra-domain reduce failed: " + nccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        if (!rccl_success)
        {
            last_error_ = "RCCL intra-domain reduce failed: " + rccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Phase1: Intra-domain reduce complete");
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 3: Intra-Domain Broadcast
    // ═══════════════════════════════════════════════════════════════════════════

    HeterogeneousBackend::Phase3Plan HeterogeneousBackend::planPhase3() const
    {
        Phase3Plan plan;
        plan.nccl_device_count = cuda_devices_.size();
        plan.rccl_device_count = rocm_devices_.size();

        // NCCL broadcast only needed if >1 CUDA device
        if (cuda_devices_.size() > 1 && nccl_backend_)
        {
            plan.will_call_nccl_broadcast = true;
            plan.nccl_broadcast_root = 0; // Bridge is always index 0 (lowest ordinal)
        }

        // RCCL broadcast only needed if >1 ROCm device
        if (rocm_devices_.size() > 1 && rccl_backend_)
        {
            plan.will_call_rccl_broadcast = true;
            plan.rccl_broadcast_root = 0; // Bridge is always index 0 (lowest ordinal)
        }

        return plan;
    }

    bool HeterogeneousBackend::executePhase3_IntraDomainBroadcast(
        const std::vector<void *> &cuda_buffers,
        const std::vector<void *> &rocm_buffers,
        size_t count,
        CollectiveDataType dtype)
    {
        if (!initialized_)
        {
            last_error_ = "HeterogeneousBackend not initialized";
            LOG_ERROR(last_error_);
            return false;
        }

        // Validate buffer counts match device counts
        if (cuda_buffers.size() != cuda_devices_.size())
        {
            last_error_ = "CUDA buffer count (" + std::to_string(cuda_buffers.size()) +
                          ") does not match CUDA device count (" +
                          std::to_string(cuda_devices_.size()) + ")";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        if (rocm_buffers.size() != rocm_devices_.size())
        {
            last_error_ = "ROCm buffer count (" + std::to_string(rocm_buffers.size()) +
                          ") does not match ROCm device count (" +
                          std::to_string(rocm_devices_.size()) + ")";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        Phase3Plan plan = planPhase3();
        bool nccl_success = true;
        bool rccl_success = true;
        std::string nccl_error;
        std::string rccl_error;

        // Execute NCCL and RCCL broadcasts in parallel using std::thread
        // They operate on independent hardware domains, so this is safe
        std::thread nccl_thread;
        std::thread rccl_thread;

        if (plan.will_call_nccl_broadcast)
        {
            LOG_DEBUG("HeterogeneousBackend Phase3: Starting NCCL broadcast from root="
                      << plan.nccl_broadcast_root << " to " << cuda_buffers.size() << " CUDA devices");
            nccl_thread = std::thread([&]()
                                      {
                // Cast away const for the buffer vector (broadcastMulti takes non-const)
                std::vector<void*> mutable_buffers(cuda_buffers.begin(), cuda_buffers.end());
                nccl_success = nccl_backend_->broadcastMulti(
                    mutable_buffers, count, dtype, plan.nccl_broadcast_root);
                if (!nccl_success) {
                    nccl_error = nccl_backend_->lastError();
                } });
        }

        if (plan.will_call_rccl_broadcast)
        {
            LOG_DEBUG("HeterogeneousBackend Phase3: Starting RCCL broadcast from root="
                      << plan.rccl_broadcast_root << " to " << rocm_buffers.size() << " ROCm devices");
            rccl_thread = std::thread([&]()
                                      {
                // Cast away const for the buffer vector (broadcastMulti takes non-const)
                std::vector<void*> mutable_buffers(rocm_buffers.begin(), rocm_buffers.end());
                rccl_success = rccl_backend_->broadcastMulti(
                    mutable_buffers, count, dtype, plan.rccl_broadcast_root);
                if (!rccl_success) {
                    rccl_error = rccl_backend_->lastError();
                } });
        }

        // Wait for both threads to complete
        if (nccl_thread.joinable())
        {
            nccl_thread.join();
        }
        if (rccl_thread.joinable())
        {
            rccl_thread.join();
        }

        // Check results
        if (!nccl_success)
        {
            last_error_ = "NCCL intra-domain broadcast failed: " + nccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        if (!rccl_success)
        {
            last_error_ = "RCCL intra-domain broadcast failed: " + rccl_error;
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend Phase3: Intra-domain broadcast complete");
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Standard 3-Phase Allreduce Helper
    // ═══════════════════════════════════════════════════════════════════════════

    bool HeterogeneousBackend::executeStandard3PhaseAllreduce(
        const std::vector<void *> &cuda_buffers,
        const std::vector<void *> &rocm_buffers,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op)
    {
        // This is the standard 3-phase pattern used as fallback when optimized
        // patterns (reduce-scatter, pipelining) are not applicable.
        //
        // Phase 1: Intra-domain reduce (each domain reduces to its bridge device)
        // Phase 2: Bridge exchange (PCIeBAR allreduce between bridge devices)
        // Phase 3: Intra-domain broadcast (broadcast result to all devices)

        LOG_DEBUG("HeterogeneousBackend: Executing standard 3-phase allreduce for "
                  << count << " elements (" << cuda_buffers.size() << " CUDA + "
                  << rocm_buffers.size() << " ROCm devices)");

        // Phase 1: Intra-domain reduce
        if (!executePhase1_IntraDomainReduce(cuda_buffers, rocm_buffers, count, dtype, op))
        {
            last_error_ = "Standard 3-phase: Phase 1 (intra-domain reduce) failed";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        // Phase 2: Bridge exchange via PCIeBAR
        // Bridge devices are cuda_buffers[0] and rocm_buffers[0]
        void *cuda_bridge_buf = cuda_buffers[0];
        void *rocm_bridge_buf = rocm_buffers[0];

        if (!executePhase2_BridgeExchange(cuda_bridge_buf, rocm_bridge_buf, count, dtype, op))
        {
            last_error_ = "Standard 3-phase: Phase 2 (bridge exchange) failed";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        // Phase 3: Intra-domain broadcast
        if (!executePhase3_IntraDomainBroadcast(cuda_buffers, rocm_buffers, count, dtype))
        {
            last_error_ = "Standard 3-phase: Phase 3 (intra-domain broadcast) failed";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        LOG_DEBUG("HeterogeneousBackend: Standard 3-phase allreduce complete");
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Synchronization
    // ═══════════════════════════════════════════════════════════════════════════

    bool HeterogeneousBackend::synchronize()
    {
        if (!initialized_)
        {
            return false;
        }

        bool success = true;

        // Synchronize all sub-backends
        if (nccl_backend_)
        {
            success &= nccl_backend_->synchronize();
        }

        if (rccl_backend_)
        {
            success &= rccl_backend_->synchronize();
        }

        if (bridge_backend_)
        {
            success &= bridge_backend_->synchronize();
        }

        return success;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Device Grouping (Private)
    // ═══════════════════════════════════════════════════════════════════════════

    bool HeterogeneousBackend::validateGroup(const DeviceGroup &group)
    {
        // Must have at least one device
        if (group.devices.empty())
        {
            last_error_ = "Device group is empty";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        // Count device types
        int cuda_count = 0;
        int rocm_count = 0;

        for (const auto &device : group.devices)
        {
            if (device.is_cuda())
            {
                cuda_count++;
            }
            else if (device.is_rocm())
            {
                rocm_count++;
            }
            else
            {
                last_error_ = "HeterogeneousBackend only supports CUDA and ROCm devices, got: " +
                              device.toString();
                LOG_ERROR("HeterogeneousBackend: " << last_error_);
                return false;
            }
        }

        // Must have at least one CUDA device
        if (cuda_count == 0)
        {
            last_error_ = "HeterogeneousBackend requires at least one CUDA device";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        // Must have at least one ROCm device
        if (rocm_count == 0)
        {
            last_error_ = "HeterogeneousBackend requires at least one ROCm device";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            return false;
        }

        return true;
    }

    bool HeterogeneousBackend::partitionDevices(const DeviceGroup &group)
    {
        cuda_devices_.clear();
        rocm_devices_.clear();

        for (const auto &device : group.devices)
        {
            if (device.is_cuda())
            {
                cuda_devices_.push_back(device);
            }
            else if (device.is_rocm())
            {
                rocm_devices_.push_back(device);
            }
        }

        // Sort by ordinal to ensure consistent ordering
        auto ordinal_cmp = [](const DeviceId &a, const DeviceId &b)
        {
            return a.ordinal < b.ordinal;
        };

        std::sort(cuda_devices_.begin(), cuda_devices_.end(), ordinal_cmp);
        std::sort(rocm_devices_.begin(), rocm_devices_.end(), ordinal_cmp);

        return true;
    }

    void HeterogeneousBackend::selectBridgeDevices()
    {
        // V1: Bridge devices are always the lowest ordinal of each type
        // This ensures deterministic behavior across runs

        if (!cuda_devices_.empty())
        {
            // Already sorted, first is lowest ordinal
            cuda_bridge_ = cuda_devices_.front();
        }

        if (!rocm_devices_.empty())
        {
            // Already sorted, first is lowest ordinal
            rocm_bridge_ = rocm_devices_.front();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Sub-Backend Management (Private)
    // ═══════════════════════════════════════════════════════════════════════════

    bool HeterogeneousBackend::createNCCLBackend()
    {
        // Only create NCCL backend if we have >1 CUDA device
        if (cuda_devices_.size() <= 1)
        {
            LOG_DEBUG("HeterogeneousBackend: Single CUDA device, skipping NCCL backend");
            nccl_backend_.reset();
            return true;
        }

        LOG_INFO("HeterogeneousBackend: Creating NCCL backend for "
                 << cuda_devices_.size() << " CUDA devices");

        // Create NCCL backend
        nccl_backend_ = std::make_unique<NCCLBackend>();

        // Build a CUDA-only device group
        DeviceGroupBuilder builder;
        builder.setName("cuda_domain")
            .setScope(CollectiveScope::LOCAL)
            .addDevices(cuda_devices_)
            .setLocalRank(0); // TODO: Proper local rank mapping

        DeviceGroup cuda_group = builder.build();

        if (!nccl_backend_->initialize(cuda_group))
        {
            last_error_ = "Failed to initialize NCCL backend";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            nccl_backend_.reset();
            return false;
        }

        LOG_INFO("HeterogeneousBackend: NCCL backend initialized");
        return true;
    }

    bool HeterogeneousBackend::createRCCLBackend()
    {
        // Only create RCCL backend if we have >1 ROCm device
        if (rocm_devices_.size() <= 1)
        {
            LOG_DEBUG("HeterogeneousBackend: Single ROCm device, skipping RCCL backend");
            rccl_backend_.reset();
            return true;
        }

        LOG_INFO("HeterogeneousBackend: Creating RCCL backend for "
                 << rocm_devices_.size() << " ROCm devices");

        // Create RCCL backend
        rccl_backend_ = std::make_unique<RCCLBackend>();

        // Build a ROCm-only device group
        DeviceGroupBuilder builder;
        builder.setName("rocm_domain")
            .setScope(CollectiveScope::LOCAL)
            .addDevices(rocm_devices_)
            .setLocalRank(0); // TODO: Proper local rank mapping

        DeviceGroup rocm_group = builder.build();

        if (!rccl_backend_->initialize(rocm_group))
        {
            last_error_ = "Failed to initialize RCCL backend";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            rccl_backend_.reset();
            return false;
        }

        LOG_INFO("HeterogeneousBackend: RCCL backend initialized");
        return true;
    }

    bool HeterogeneousBackend::createBridgeBackend()
    {
        LOG_INFO("HeterogeneousBackend: Creating PCIeBAR bridge backend for "
                 << cuda_bridge_.toString() << " <-> " << rocm_bridge_.toString());

        // Create PCIeBAR backend
        bridge_backend_ = std::make_unique<PCIeBARBackend>();

        // Build a bridge-only device group (just the two bridge devices)
        DeviceGroupBuilder builder;
        builder.setName("pcie_bridge")
            .setScope(CollectiveScope::LOCAL)
            .addDevice(cuda_bridge_)
            .addDevice(rocm_bridge_)
            .setLocalRank(0); // Bridge operates from CUDA perspective

        DeviceGroup bridge_group = builder.build();

        if (!bridge_backend_->initialize(bridge_group))
        {
            last_error_ = "Failed to initialize PCIeBAR bridge backend";
            LOG_ERROR("HeterogeneousBackend: " << last_error_);
            bridge_backend_.reset();
            return false;
        }

        LOG_INFO("HeterogeneousBackend: PCIeBAR bridge backend initialized");
        return true;
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2
