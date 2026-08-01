/**
 * @file MoEExpertWeightService.cpp
 * @brief Implementation of MoE expert weight lifecycle service.
 *
 * Contains all weight preparation, serialization, and rebalancing logic
 * extracted from MoEExpertComputeStage.cpp. See MoEExpertWeightService.h for API docs.
 */

#include "MoEExpertWeightService.h"
#include "ExpertWeightPayloadProvider.h"
#include "GPUExpertTransfer.h"
#include "GpuExpertSlotPool.h"
#include "GpuExpertTransferStagingPool.h"
#include "DeviceMoERebalanceController.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/BlockStructures.h"
#include "../../kernels/KernelFactory.h"
#include "../../kernels/PackedWeightsSerialization.h"
#include "../../kernels/cpu/gemm/CPUPackedWeights.h"
#include "../../kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "../../loaders/MmapRegion.h"
#include "../../loaders/ExpertGemmRegistry.h"
#include "../../loaders/GPUVramPreflight.h"
#include "../../loaders/gpu_pipeline/LoadOrchestrator.h"
#include "../../backends/BackendManager.h"
#include "../../backends/GPUDeviceContextPool.h"
#include "../../utils/Assertions.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include "../../utils/OpenMPUtils.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../loaders/PreparedWeightStore.h"

#ifdef HAVE_CUDA
#include "../../kernels/cuda/gemm/CUDAWeightPacker.h"
#include "../../kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif

#ifdef HAVE_ROCM
#include "../../kernels/rocm/gemm/ROCmWeightPacker.h"
#include "../../kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#endif

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#include <numa.h>
#include <numaif.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // Alias for fully-qualified KernelFactory access
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    namespace
    {

        /// Query the NUMA node of a virtual address using move_pages(2).
        /// Returns -1 if NUMA info unavailable (non-Linux or unmapped page).
        static int queryNUMANode(const void *ptr)
        {
#ifdef __linux__
            if (!ptr)
                return -1;
            void *pages[] = {const_cast<void *>(ptr)};
            int status[1] = {-1};
            if (move_pages(0, 1, pages, nullptr, status, 0) == 0 && status[0] >= 0)
                return status[0];
#endif
            (void)ptr;
            return -1;
        }

        static std::shared_ptr<void> reusableGpuDirectTransferStreamFor(
            const DeviceId &device,
            uint64_t &create_ns,
            int &created_count,
            int &reuse_count)
        {
            if (!device.is_gpu())
                return nullptr;

            bool created = false;
            const auto stream_start = std::chrono::steady_clock::now();
            IWorkerGPUContext &gpu_ctx = GPUDeviceContextPool::instance().getContext(device);
            void *stream = gpu_ctx.getOrCreateAuxiliaryStream("moe_gpu_direct_transfer", &created);
            create_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - stream_start)
                                                   .count());
            if (!stream)
                return nullptr;

            if (created)
                ++created_count;
            else
                ++reuse_count;

            return std::shared_ptr<void>(stream, [](void *) {});
        }

        static size_t systemPageSize()
        {
#ifdef __linux__
            long page_size = sysconf(_SC_PAGESIZE);
            if (page_size > 0)
                return static_cast<size_t>(page_size);
#endif
            return 4096;
        }

        static bool verifyRangeNUMANode(const void *ptr, size_t bytes, int expected_node, const char *label)
        {
            if (!ptr || bytes == 0)
                return true;

            const size_t page_size = systemPageSize();
            const uintptr_t raw_start = reinterpret_cast<uintptr_t>(ptr);
            const uintptr_t raw_end = raw_start + bytes;
            const uintptr_t page_start = raw_start & ~(static_cast<uintptr_t>(page_size) - 1);
            const uintptr_t last_page = (raw_end - 1) & ~(static_cast<uintptr_t>(page_size) - 1);
            const uintptr_t mid_page = ((page_start + last_page) / 2) & ~(static_cast<uintptr_t>(page_size) - 1);

            uintptr_t sample_pages[] = {page_start, mid_page, last_page};
            for (uintptr_t page : sample_pages)
            {
                const int node = queryNUMANode(reinterpret_cast<const void *>(page));
                if (node < 0)
                {
                    LOG_ERROR("[MoEWeightService][NUMA] Cannot verify NUMA page placement for " << label
                                                                                                 << " at " << reinterpret_cast<const void *>(page));
                    return false;
                }
                if (node != expected_node)
                {
                    LOG_ERROR("[MoEWeightService][NUMA] NUMA migration verification failed for " << label
                                                                                                  << ": expected node " << expected_node
                                                                                                  << ", found node " << node
                                                                                                  << " at " << reinterpret_cast<const void *>(page));
                    return false;
                }
            }

            return true;
        }

        static bool migrateRangeToNUMANode(void *ptr, size_t bytes, int target_node, const char *label)
        {
#ifdef __linux__
            if (!ptr || bytes == 0)
                return true;
            if (target_node < 0)
            {
                LOG_ERROR("[MoEWeightService][NUMA] Cannot migrate " << label
                                                                      << ": target NUMA node is unknown");
                return false;
            }
            if (numa_available() < 0)
            {
                LOG_ERROR("[MoEWeightService][NUMA] Cannot migrate " << label
                                                                      << ": libnuma policy APIs are unavailable");
                return false;
            }
            if (target_node > numa_max_node())
            {
                LOG_ERROR("[MoEWeightService][NUMA] Cannot migrate " << label
                                                                      << ": target NUMA node " << target_node
                                                                      << " exceeds max node " << numa_max_node());
                return false;
            }

            const size_t page_size = systemPageSize();
            const uintptr_t raw_start = reinterpret_cast<uintptr_t>(ptr);
            const uintptr_t raw_end = raw_start + bytes;
            const uintptr_t page_start = raw_start & ~(static_cast<uintptr_t>(page_size) - 1);
            const uintptr_t page_end = (raw_end + page_size - 1) & ~(static_cast<uintptr_t>(page_size) - 1);
            const size_t page_bytes = static_cast<size_t>(page_end - page_start);

            struct bitmask *nodemask = numa_allocate_nodemask();
            if (!nodemask)
            {
                LOG_ERROR("[MoEWeightService][NUMA] Failed to allocate nodemask for " << label);
                return false;
            }

            numa_bitmask_clearall(nodemask);
            numa_bitmask_setbit(nodemask, target_node);
            errno = 0;
            const int rc = mbind(reinterpret_cast<void *>(page_start),
                                 page_bytes,
                                 MPOL_BIND,
                                 nodemask->maskp,
                                 nodemask->size,
                                 MPOL_MF_MOVE | MPOL_MF_STRICT);
            const int bind_errno = errno;
            numa_free_nodemask(nodemask);

            if (rc != 0)
            {
                LOG_WARN("[MoEWeightService][NUMA] mbind migration failed for " << label
                                                                                << " (" << page_bytes
                                                                                << " page-rounded bytes, node="
                                                                                << target_node << "): errno="
                                                                                << bind_errno << " ("
                                                                                << std::strerror(bind_errno) << ")");
                return false;
            }

            return verifyRangeNUMANode(ptr, bytes, target_node, label);
#else
            (void)ptr;
            (void)bytes;
            (void)target_node;
            (void)label;
            return false;
#endif
        }

        /**
         * @brief Bind a fresh page-aligned allocation to a NUMA node before
         *        first touch.
         *
         * Strict migration is still the preferred fast path for already-packed
         * NativeVNNI buffers, because it preserves the single packed allocation.
         * When Linux refuses that post-hoc migration, the fix-forward path is
         * to allocate a replacement buffer, install the target-node memory
         * policy with no migration flags, and then copy the bytes into the
         * policy-bound pages.  This preserves the optimized grouped CPU
         * implementation and still refuses to run if target placement cannot be
         * proven.
         */
        static bool bindFreshRangePolicyToNUMANode(void *ptr,
                                                   size_t bytes,
                                                   int target_node,
                                                   const char *label)
        {
#ifdef __linux__
            if (!ptr || bytes == 0)
                return true;
            if (target_node < 0)
            {
                LOG_ERROR("[MoEWeightService][NUMA] Cannot bind fresh range for " << label
                                                                                  << ": target NUMA node is unknown");
                return false;
            }
            if (numa_available() < 0)
            {
                LOG_ERROR("[MoEWeightService][NUMA] Cannot bind fresh range for " << label
                                                                                  << ": libnuma policy APIs are unavailable");
                return false;
            }
            if (target_node > numa_max_node())
            {
                LOG_ERROR("[MoEWeightService][NUMA] Cannot bind fresh range for " << label
                                                                                  << ": target NUMA node " << target_node
                                                                                  << " exceeds max node " << numa_max_node());
                return false;
            }

            const size_t page_size = systemPageSize();
            const uintptr_t raw_start = reinterpret_cast<uintptr_t>(ptr);
            if ((raw_start % page_size) != 0)
            {
                LOG_ERROR("[MoEWeightService][NUMA] Fresh range for " << label
                                                                      << " is not page-aligned; strict NUMA placement would "
                                                                         "need to include neighboring heap pages");
                return false;
            }
            const size_t page_bytes = (bytes + page_size - 1) & ~(page_size - 1);

            struct bitmask *nodemask = numa_allocate_nodemask();
            if (!nodemask)
            {
                LOG_ERROR("[MoEWeightService][NUMA] Failed to allocate nodemask for fresh range " << label);
                return false;
            }

            numa_bitmask_clearall(nodemask);
            numa_bitmask_setbit(nodemask, target_node);
            errno = 0;
            const int rc = mbind(ptr,
                                 page_bytes,
                                 MPOL_BIND,
                                 nodemask->maskp,
                                 nodemask->size,
                                 0);
            const int bind_errno = errno;
            numa_free_nodemask(nodemask);

            if (rc != 0)
            {
                LOG_ERROR("[MoEWeightService][NUMA] mbind fresh-range policy failed for "
                          << label << " (" << page_bytes << " page-rounded bytes, node="
                          << target_node << "): errno=" << bind_errno << " ("
                          << std::strerror(bind_errno) << ")");
                return false;
            }

            return true;
#else
            (void)ptr;
            (void)bytes;
            (void)target_node;
            (void)label;
            return false;
#endif
        }

        /**
         * @brief Verify every page of a repaired packed-weight buffer.
         *
         * Normal NUMA audits sample a few pages to avoid adding many syscalls
         * to common expert preparation.  The relocation path only runs after
         * strict migration has failed, so it pays the extra move_pages(2) calls
         * to prove the replacement buffer is completely resident on the target
         * node before grouped CPU decode can consume it.
         */
        static bool verifyEveryPageNUMANode(const void *ptr,
                                            size_t bytes,
                                            int expected_node,
                                            const char *label)
        {
            if (!ptr || bytes == 0)
                return true;

            const size_t page_size = systemPageSize();
            const uintptr_t raw_start = reinterpret_cast<uintptr_t>(ptr);
            const uintptr_t raw_end = raw_start + bytes;
            const uintptr_t page_start = raw_start & ~(static_cast<uintptr_t>(page_size) - 1);
            const uintptr_t page_end = (raw_end + page_size - 1) & ~(static_cast<uintptr_t>(page_size) - 1);

            for (uintptr_t page = page_start; page < page_end; page += page_size)
            {
                const int node = queryNUMANode(reinterpret_cast<const void *>(page));
                if (node < 0)
                {
                    LOG_ERROR("[MoEWeightService][NUMA] Cannot verify relocated NUMA page for "
                              << label << " at " << reinterpret_cast<const void *>(page));
                    return false;
                }
                if (node != expected_node)
                {
                    LOG_ERROR("[MoEWeightService][NUMA] Relocated NUMA verification failed for "
                              << label << ": expected node " << expected_node
                              << ", found node " << node
                              << " at " << reinterpret_cast<const void *>(page));
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Replace a packed NativeVNNI buffer with a target-node-local
         *        copy when Linux refuses strict post-pack migration.
         */
        static bool relocatePackedInterleavedToNUMANode(
            cpu::native_vnni::CPUNativeVNNIPackedWeights &packed,
            int target_node,
            const std::string &label)
        {
            const size_t bytes = packed.native_interleaved.size();
            if (bytes == 0)
                return true;

            AlignedVector<uint8_t> relocated;
            try
            {
                relocated.resize_uninitialized(bytes);
            }
            catch (const std::bad_alloc &)
            {
                LOG_ERROR("[MoEWeightService][NUMA] Failed to allocate relocation buffer for "
                          << label << " (" << bytes << " bytes)");
                return false;
            }

            const std::string relocation_label = label + " relocated_native_interleaved";
            if (!bindFreshRangePolicyToNUMANode(
                    relocated.data(),
                    relocated.size(),
                    target_node,
                    relocation_label.c_str()))
            {
                return false;
            }

            std::memcpy(relocated.data(), packed.native_interleaved.data(), bytes);
            if (!verifyEveryPageNUMANode(
                    relocated.data(),
                    relocated.size(),
                    target_node,
                    relocation_label.c_str()))
            {
                return false;
            }

            packed.native_interleaved.swap(relocated);
            packed.clearWorkspace();
            return true;
        }

        static bool enforceExpertKernelNUMA(ITensorGemm *kernel,
                                            int target_node,
                                            int layer_idx,
                                            int expert_id,
                                            const char *role)
        {
            auto *vnni_kernel = dynamic_cast<cpu::native_vnni::CPUNativeVNNIGemmKernel *>(kernel);
            if (!vnni_kernel)
                return true;

            auto &packed = const_cast<cpu::native_vnni::CPUNativeVNNIPackedWeights &>(
                vnni_kernel->packedWeights());

            std::ostringstream interleaved_label;
            interleaved_label << "layer " << layer_idx << " expert " << expert_id
                              << " " << role << " native_interleaved";
            if (!migrateRangeToNUMANode(packed.native_interleaved.data(),
                                        packed.native_interleaved.size(),
                                        target_node,
                                        interleaved_label.str().c_str()))
            {
                return relocatePackedInterleavedToNUMANode(
                    packed,
                    target_node,
                    interleaved_label.str());
            }

            return true;
        }

        /// Audit NUMA placement of expert GEMM weights.
        static void auditExpertNUMA(
            const std::vector<int> &experts,
            const std::vector<ITensorGemm *> &gate_gemms,
            const char *label,
            int layer_idx,
            int expected_node,
            int max_sample = 8)
        {
            if (experts.empty())
                return;

            if (expected_node < 0)
            {
                LOG_DEBUG("[MoEWeightService][NUMA] Cannot determine expected NUMA node, skipping audit");
                return;
            }

            int sampled = 0, local = 0, remote = 0, unmapped = 0;
            int first_remote_expert = -1;
            int first_remote_node = -1;

            const int step = (max_sample > 0 && static_cast<int>(experts.size()) > max_sample)
                                 ? static_cast<int>(experts.size()) / max_sample
                                 : 1;

            for (size_t i = 0; i < experts.size(); i += step)
            {
                const int e = experts[i];
                if (e < 0 || e >= static_cast<int>(gate_gemms.size()) || !gate_gemms[e])
                    continue;

                auto *vnni_kernel = dynamic_cast<const cpu::native_vnni::CPUNativeVNNIGemmKernel *>(gate_gemms[e]);
                if (!vnni_kernel)
                    continue;

                const auto &packed = vnni_kernel->packedWeights();
                const void *data_ptr = packed.interleavedBase();
                if (!data_ptr)
                    continue;

                int node = queryNUMANode(data_ptr);
                ++sampled;

                if (node < 0)
                {
                    ++unmapped;
                }
                else if (node != expected_node)
                {
                    ++remote;
                    if (first_remote_expert < 0)
                    {
                        first_remote_expert = e;
                        first_remote_node = node;
                    }
                }
                else
                {
                    ++local;
                }
            }

            if (sampled == 0)
                return;

            if (remote > 0)
            {
                LOG_WARN("[MoEWeightService][NUMA] layer " << layer_idx << " " << label
                                                           << ": " << remote << "/" << sampled << " sampled experts on WRONG NUMA node"
                                                           << " (expected=" << expected_node
                                                           << ", first_remote: expert " << first_remote_expert
                                                           << " on node " << first_remote_node << ")"
                                                           << (unmapped > 0 ? (std::string(" [") + std::to_string(unmapped) + " unmapped]") : ""));
            }
            else
            {
                LOG_DEBUG("[MoEWeightService][NUMA] layer " << layer_idx << " " << label
                                                            << ": all " << sampled << "/" << static_cast<int>(experts.size())
                                                            << " sampled experts on correct NUMA node " << expected_node
                                                            << (unmapped > 0 ? (std::string(" [") + std::to_string(unmapped) + " unmapped]") : ""));
            }
        }

        static size_t quantizedViewRawBytes(const TensorBase &tensor)
        {
            const size_t reported = tensor.size_bytes();
            if (reported > 0)
                return reported;

            const auto &shape = tensor.shape();
            if (shape.size() != 2)
                return 0;

            const size_t rows = shape[0];
            const size_t cols = shape[1];
            auto bytes_for = [rows, cols](size_t block_size, size_t block_bytes) -> size_t
            {
                const size_t blocks_per_row = (cols + block_size - 1) / block_size;
                return rows * blocks_per_row * block_bytes;
            };

            switch (tensor.native_type())
            {
            case TensorType::IQ4_NL:
                return bytes_for(IQ4_NLBlock::BLOCK_SIZE, sizeof(IQ4_NLBlock));
            case TensorType::IQ4_XS:
                return bytes_for(IQ4_XSBlock::BLOCK_SIZE, sizeof(IQ4_XSBlock));
            case TensorType::Q8_0:
                return bytes_for(Q8_0Block::BLOCK_SIZE, sizeof(Q8_0Block));
            case TensorType::Q4_0:
                return bytes_for(Q4_0Block::BLOCK_SIZE, sizeof(Q4_0Block));
            case TensorType::Q4_1:
                return bytes_for(Q4_1Block::BLOCK_SIZE, sizeof(Q4_1Block));
            case TensorType::Q5_0:
                return bytes_for(Q5_0Block::BLOCK_SIZE, sizeof(Q5_0Block));
            case TensorType::Q5_1:
                return bytes_for(Q5_1Block::BLOCK_SIZE, sizeof(Q5_1Block));
            case TensorType::Q2_K:
                return bytes_for(Q2_KBlock::BLOCK_SIZE, sizeof(Q2_KBlock));
            case TensorType::Q3_K:
                return bytes_for(Q3_KBlock::BLOCK_SIZE, sizeof(Q3_KBlock));
            case TensorType::Q4_K:
                return bytes_for(Q4_KBlock::BLOCK_SIZE, sizeof(Q4_KBlock));
            case TensorType::Q5_K:
                return bytes_for(Q5_KBlock::BLOCK_SIZE, sizeof(Q5_KBlock));
            case TensorType::Q6_K:
                return bytes_for(Q6_KBlock::BLOCK_SIZE, sizeof(Q6_KBlock));
            case TensorType::Q8_K:
                return bytes_for(Q8_KBlock::BLOCK_SIZE, sizeof(Q8_KBlock));
            case TensorType::IQ2_XXS:
                return bytes_for(IQ2_XXSBlock::BLOCK_SIZE, sizeof(IQ2_XXSBlock));
            case TensorType::IQ2_XS:
                return bytes_for(IQ2_XSBlock::BLOCK_SIZE, sizeof(IQ2_XSBlock));
            case TensorType::IQ2_S:
                return bytes_for(IQ2_SBlock::BLOCK_SIZE, sizeof(IQ2_SBlock));
            case TensorType::IQ3_XXS:
                return bytes_for(IQ3_XXSBlock::BLOCK_SIZE, sizeof(IQ3_XXSBlock));
            case TensorType::IQ3_S:
                return bytes_for(IQ3_SBlock::BLOCK_SIZE, sizeof(IQ3_SBlock));
            case TensorType::IQ1_S:
                return bytes_for(IQ1_SBlock::BLOCK_SIZE, sizeof(IQ1_SBlock));
            case TensorType::IQ1_M:
                return bytes_for(IQ1_MBlock::BLOCK_SIZE, sizeof(IQ1_MBlock));
            default:
                return 0;
            }
        }

        static ExpertSlabDescriptor makeExpertSlabDescriptor(const MoEWeightContext &ctx, WeightRole role)
        {
            ExpertSlabDescriptor desc;
            desc.layer_idx = ctx.layer_idx;
            desc.role = role;
            desc.device = ctx.device_id;
            desc.num_experts = ctx.num_experts;
            desc.local_expert_start = ctx.local_expert_start;
            desc.local_expert_count = (ctx.local_expert_count < 0) ? ctx.num_experts : ctx.local_expert_count;
            if (role == WeightRole::MoEExpertDown)
            {
                desc.rows_per_expert = ctx.d_model;
                desc.cols_per_expert = ctx.expert_intermediate;
            }
            else
            {
                desc.rows_per_expert = ctx.expert_intermediate;
                desc.cols_per_expert = ctx.d_model;
            }
            return desc;
        }

        static std::optional<ExpertSlabRef> resolveExpertSlabRef(
            MoEWeightContext &ctx,
            WeightRole role,
            std::optional<ExpertSlabRef> &cached_ref,
            bool create_if_missing)
        {
            if (!ctx.prepared_store)
                return std::nullopt;
            if (cached_ref.has_value())
                return cached_ref;

            const auto desc = makeExpertSlabDescriptor(ctx, role);
            auto found = ctx.prepared_store->findExpertSlab(desc);
            if (found.has_value())
            {
                cached_ref = *found;
                return cached_ref;
            }

            if (!create_if_missing)
                return std::nullopt;

            cached_ref = ctx.prepared_store->registerExpertSlab(desc);
            return cached_ref;
        }

        static void resolveExpertSlabRefs(MoEWeightContext &ctx, bool create_if_missing)
        {
            (void)resolveExpertSlabRef(ctx, WeightRole::MoEExpertGate, ctx.gate_slab_ref, create_if_missing);
            (void)resolveExpertSlabRef(ctx, WeightRole::MoEExpertUp, ctx.up_slab_ref, create_if_missing);
            (void)resolveExpertSlabRef(ctx, WeightRole::MoEExpertDown, ctx.down_slab_ref, create_if_missing);
        }

        static int gpuOrdinalFor(const DeviceId &device)
        {
            if (device.is_cuda())
                return device.cuda_ordinal();
            if (device.is_rocm())
                return device.rocm_ordinal();
            return -1;
        }

        static const char *gpuDirectRoleLabel(WeightRole role)
        {
            if (role == WeightRole::MoEExpertGate)
                return "gate";
            if (role == WeightRole::MoEExpertUp)
                return "up";
            if (role == WeightRole::MoEExpertDown)
                return "down";
            return nullptr;
        }

        static DeviceNativeVNNIMatrixDesc nativeVnniDescFromSlot(
            const WeightVRAMPool::WeightSlot &slot,
            int N,
            int K,
            uint32_t blocks_per_row,
            uint8_t codebook_id)
        {
            DeviceNativeVNNIMatrixDesc desc{};
            desc.payload = slot.d_native_vnni_payload;
            desc.scales = slot.d_native_vnni_scales;
            desc.mins = slot.d_native_vnni_mins;
            desc.emins = slot.d_native_vnni_emins;
            desc.n = N;
            desc.k = K;
            desc.blocks_per_row = blocks_per_row;
            desc.codebook_id = codebook_id;
            return desc;
        }

        static GpuExpertPackedDescriptor packedDescFromSlot(
            const WeightVRAMPool::WeightSlot &slot,
            int N,
            int K,
            uint32_t blocks_per_row,
            uint8_t codebook_id,
            uint8_t payload_bytes_per_block,
            bool is_asymmetric,
            bool has_emins)
        {
            return makeGpuExpertPackedDescriptor(
                nativeVnniDescFromSlot(slot, N, K, blocks_per_row, codebook_id),
                payload_bytes_per_block,
                is_asymmetric,
                has_emins);
        }

        static std::shared_ptr<ITensorGemm> createGpuDirectKernelFromSlot(
            const DeviceId &device,
            int device_ordinal,
            const WeightVRAMPool::WeightSlot &slot,
            int N,
            int K,
            uint32_t blocks_per_row,
            uint8_t codebook_id,
            const std::shared_ptr<void> &lifetime_owner)
        {
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                return std::make_shared<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                    N, K, device_ordinal,
                    slot.d_native_vnni_payload,
                    static_cast<uint16_t *>(slot.d_native_vnni_scales),
                    static_cast<uint16_t *>(slot.d_native_vnni_mins),
                    static_cast<uint32_t *>(slot.d_native_vnni_emins),
                    codebook_id, blocks_per_row,
                    lifetime_owner);
            }
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                return std::make_shared<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                    N, K, device_ordinal,
                    slot.d_native_vnni_payload,
                    slot.d_native_vnni_scales,
                    slot.d_native_vnni_mins,
                    slot.d_native_vnni_emins,
                    codebook_id, blocks_per_row,
                    lifetime_owner);
            }
#endif
            return nullptr;
        }

        static std::shared_ptr<void> eventOwnerFor(IBackend *backend, int ordinal, void *event)
        {
            return std::shared_ptr<void>(
                event,
                [backend, ordinal](void *ptr)
                {
                    if (ptr && backend)
                        backend->destroyEvent(ptr, ordinal);
                });
        }

        static void appendUniqueLifetime(
            std::vector<std::shared_ptr<void>> &lifetimes,
            const std::shared_ptr<void> &lifetime)
        {
            if (!lifetime)
                return;
            auto it = std::find(lifetimes.begin(), lifetimes.end(), lifetime);
            if (it == lifetimes.end())
                lifetimes.push_back(lifetime);
        }

        static std::shared_ptr<std::mutex> gpuInitialExpertPrepareMutexFor(const MoEWeightContext &ctx)
        {
            static std::mutex registry_mutex;
            static std::unordered_map<std::string, std::weak_ptr<std::mutex>> registry;

            std::ostringstream key;
            key << static_cast<const void *>(ctx.prepared_store)
                << '|' << ctx.device_id.to_string()
                << "|layer=" << ctx.layer_idx
                << "|experts=" << ctx.num_experts
                << "|local=" << ctx.local_expert_start
                << ':' << ((ctx.local_expert_count < 0) ? ctx.num_experts : ctx.local_expert_count)
                << "|shape=" << ctx.expert_intermediate
                << 'x' << ctx.d_model;

            std::lock_guard<std::mutex> lock(registry_mutex);
            auto &entry = registry[key.str()];
            if (auto existing = entry.lock())
                return existing;

            auto created = std::make_shared<std::mutex>();
            entry = created;
            return created;
        }

    } // anonymous namespace

    // =========================================================================
    // extractExpertViews — Create 2D views from 3D packed tensors
    // =========================================================================

    bool MoEExpertWeightService::extractExpertViews(MoEWeightContext &ctx)
    {
        if (!ctx.gate_exps || !ctx.up_exps || !ctx.down_exps)
        {
            LOG_ERROR("[MoEWeightService] Cannot extract views: null expert tensors");
            return false;
        }

        const int num_experts = ctx.num_experts;
        if (num_experts <= 0)
        {
            LOG_ERROR("[MoEWeightService] Invalid num_experts: " << num_experts);
            return false;
        }

        // Expert-ID ownership range: extract views only for local experts.
        // Dynamic rebalancing may set an expert mask while using replicated
        // parent tensors. In that case, extracting all views is safe because
        // every global expert id is physically present. LocalTP static expert
        // ownership can also be expressed as a mask, but its parent tensor is
        // presliced to the local expert range. Presliced tensors are the source
        // of truth for physical bounds, so they must never attempt to view
        // global expert ids outside [local_start, local_end).
        const bool extract_all = !ctx.expert_mask.empty();
        const size_t active_mask_count =
            static_cast<size_t>(std::count(ctx.expert_mask.begin(), ctx.expert_mask.end(), true));
        const int local_start = ctx.local_expert_start;
        const int local_count = (ctx.local_expert_count < 0)
                                    ? num_experts
                                    : ctx.local_expert_count;
        const int local_end = local_start + local_count;

        ctx.expert_gate_views.resize(num_experts);
        ctx.expert_up_views.resize(num_experts);
        ctx.expert_down_views.resize(num_experts);

        // Extract 2D views for each expert.
        // GGUF 3D: shape = [ne[0], ne[1], ne[2]] = [cols, rows, num_experts_in_tensor]
        // Each expert's 2D slice is [rows, cols] at element offset within the tensor.
        //
        // With expert-ID-apportioned loading, the 3D tensor may contain only
        // local experts (shape[2] == local_count) instead of all experts.
        // In that case, global expert index `e` maps to local tensor index
        // `e - local_start`. When the tensor has all experts (shape[2] == num_experts),
        // the offset uses the global index directly.
        auto extract_views = [local_start, local_count, local_end, extract_all, active_mask_count, &ctx](
                                 TensorBase *tensor_3d, int n_experts,
                                 const char *role_name,
                                 std::vector<std::shared_ptr<TensorBase>> &views) -> bool
        {
            const auto &shape = tensor_3d->shape();
            if (shape.size() != 3)
            {
                LOG_ERROR("[MoE] Expert tensor must be 3D, got " << shape.size() << "D");
                return false;
            }

            // GGUF 3D: shape[0]=ne[0]=cols (fastest), shape[1]=ne[1]=rows, shape[2]=experts in tensor
            size_t cols = shape[0];
            size_t rows = shape[1];
            size_t tensor_expert_count = shape[2];
            size_t elements_per_expert = rows * cols;

            // Determine whether the expert-ID axis was pre-sliced.
            // or contains all experts (replicated mode).
            const bool is_presliced = (static_cast<int>(tensor_expert_count) != n_experts);

            for (int e = 0; e < n_experts; ++e)
            {
                // Skip non-local expert IDs under apportioned ownership. A replicated tensor can
                // still extract every global expert when dynamic rebalance asks
                // for it, but a presliced LocalTP tensor contains only local
                // expert storage and must clamp to the physical local range.
                if ((is_presliced || !extract_all) && (e < local_start || e >= local_end))
                    continue;

                // For pre-sliced tensors: local expert `e` is at tensor index `e - local_start`
                // For full tensors: expert `e` is at tensor index `e`
                size_t tensor_idx = is_presliced ? static_cast<size_t>(e - local_start)
                                                 : static_cast<size_t>(e);
                size_t element_offset = tensor_idx * elements_per_expert;

                std::vector<size_t> view_shape = {rows, cols};
                std::shared_ptr<TensorBase> view;
                try
                {
                    view = tensor_3d->create_view(view_shape, element_offset);
                }
                catch (const std::exception &ex)
                {
                    std::ostringstream oss;
                    oss << "[MoEWeightService] create_view failed"
                        << " layer=" << ctx.layer_idx
                        << " role=" << role_name
                        << " expert=" << e
                        << " tensor_idx=" << tensor_idx
                        << " tensor_shape=[" << shape[0] << "," << shape[1] << "," << shape[2] << "]"
                        << " view_shape=[" << rows << "," << cols << "]"
                        << " offset_elements=" << element_offset
                        << " num_experts=" << n_experts
                        << " tensor_expert_count=" << tensor_expert_count
                        << " is_presliced=" << (is_presliced ? "true" : "false")
                        << " local_start=" << local_start
                        << " local_count=" << local_count
                        << " local_end=" << local_end
                        << " mask_size=" << ctx.expert_mask.size()
                        << " active_mask_count=" << active_mask_count
                        << ": " << ex.what();
                    throw std::runtime_error(oss.str());
                }
                if (!view)
                {
                    LOG_ERROR("[MoE] Failed to create view for expert " << e
                                                                        << " (tensor_idx=" << tensor_idx << ")");
                    return false;
                }
                views[e] = std::move(view);
            }
            return true;
        };

        if (!extract_views(ctx.gate_exps, num_experts, "gate", ctx.expert_gate_views))
            return false;
        if (!extract_views(ctx.up_exps, num_experts, "up", ctx.expert_up_views))
            return false;
        if (!extract_views(ctx.down_exps, num_experts, "down", ctx.expert_down_views))
            return false;

        LOG_TRACE("[MoEWeightService] Extracted " << (extract_all ? num_experts : local_count) << "/" << num_experts
                                                  << " expert 2D views (expert-ID range [" << local_start
                                                  << ", " << local_end << ")"
                                                  << (extract_all ? " extract_all=true" : "") << ")");
        return true;
    }

    // =========================================================================
    // prepareGemmEngines — Prepare GEMM engines for all expert views
    // =========================================================================

    bool MoEExpertWeightService::prepareGemmEngines(MoEWeightContext &ctx)
    {
        const int num_experts = ctx.num_experts;
        if (ctx.expert_gate_views.empty() ||
            static_cast<int>(ctx.expert_gate_views.size()) != num_experts)
        {
            LOG_ERROR("[MoEWeightService] prepareGemmEngines: call extractExpertViews() first");
            return false;
        }

        // Expert-ID ownership range.
        // Dynamic rebalancing: when expert_mask is set, prepare ONLY mask-active
        // experts (not all). Views exist for all experts, but GEMM engines are
        // expensive (VNNI repacking). Newly-acquired experts get engines from
        // serialized payloads via registerAndPrepareNewExperts() after rebalancing.
        const bool use_mask = !ctx.expert_mask.empty();
        const int local_start = ctx.local_expert_start;
        const int local_count = (ctx.local_expert_count < 0)
                                    ? num_experts
                                    : ctx.local_expert_count;
        const int local_end = local_start + local_count;

        // Build list of experts to prepare
        std::vector<int> experts_to_prep;
        if (use_mask)
        {
            for (int e = 0; e < num_experts; ++e)
                if (ctx.expert_mask[e])
                    experts_to_prep.push_back(e);
        }
        else
        {
            for (int e = local_start; e < local_end; ++e)
                experts_to_prep.push_back(e);
        }
        const std::vector<int> required_experts = experts_to_prep;
        int prep_count = static_cast<int>(experts_to_prep.size());

        ctx.prepared_gate_gemm.resize(num_experts, nullptr);
        ctx.prepared_up_gemm.resize(num_experts, nullptr);
        ctx.prepared_down_gemm.resize(num_experts, nullptr);

        std::shared_ptr<std::mutex> gpu_prepare_mutex;
        std::unique_lock<std::mutex> gpu_prepare_lock;
        if (ctx.device_id.is_gpu() && ctx.prepared_store)
        {
            /*
             * Prefill/decode graph stages for the same layer may be constructed
             * concurrently.  Hold this per-store/device/layer mutex through
             * store lookup and initial GPU pack so the second stage observes the
             * first stage's registered slabs instead of allocating another full
             * expert pool.
             */
            gpu_prepare_mutex = gpuInitialExpertPrepareMutexFor(ctx);
            gpu_prepare_lock = std::unique_lock<std::mutex>(*gpu_prepare_mutex);
        }

        if (ctx.prepared_store)
        {
            auto make_desc = [&](WeightRole role)
            {
                (void)local_count;
                return makeExpertSlabDescriptor(ctx, role);
            };

            auto has_required_experts = [&](const ExpertSlabRef &ref)
            {
                const auto available = ctx.prepared_store->expertAvailabilityMask(ref);
                if (static_cast<int>(available.size()) != num_experts)
                    return false;
                for (int e : experts_to_prep)
                {
                    if (e < 0 || e >= static_cast<int>(available.size()) || !available[e])
                        return false;
                }
                return true;
            };

            auto gate_ref = ctx.prepared_store->findExpertSlab(make_desc(WeightRole::MoEExpertGate));
            auto up_ref = ctx.prepared_store->findExpertSlab(make_desc(WeightRole::MoEExpertUp));
            auto down_ref = ctx.prepared_store->findExpertSlab(make_desc(WeightRole::MoEExpertDown));

            const bool any_existing_slab = gate_ref.has_value() || up_ref.has_value() || down_ref.has_value();
            if (any_existing_slab && (!gate_ref || !up_ref || !down_ref))
            {
                LOG_ERROR("[MoEWeightService] PreparedWeightStore has partial expert slabs for layer "
                          << ctx.layer_idx << " on " << ctx.device_id.to_string()
                          << "; refusing raw expert repack fallback");
                return false;
            }

            if (gate_ref && up_ref && down_ref &&
                has_required_experts(*gate_ref) &&
                has_required_experts(*up_ref) &&
                has_required_experts(*down_ref))
            {
                bool resolved_all = true;
                for (int e : experts_to_prep)
                {
                    ctx.prepared_gate_gemm[e] = ctx.prepared_store->expertGemmKernel(*gate_ref, e);
                    ctx.prepared_up_gemm[e] = ctx.prepared_store->expertGemmKernel(*up_ref, e);
                    ctx.prepared_down_gemm[e] = ctx.prepared_store->expertGemmKernel(*down_ref, e);
                    if (!ctx.prepared_gate_gemm[e] || !ctx.prepared_up_gemm[e] || !ctx.prepared_down_gemm[e])
                    {
                        resolved_all = false;
                        break;
                    }
                }

                if (resolved_all)
                {
                    ctx.gate_slab_ref = *gate_ref;
                    ctx.up_slab_ref = *up_ref;
                    ctx.down_slab_ref = *down_ref;
                    LOG_DEBUG("[MoEWeightService] Reused " << (prep_count * 3)
                                                           << (ctx.device_id.is_gpu() ? " GPU" : " CPU")
                                                           << " expert GEMM engines from PreparedWeightStore for layer "
                                                           << ctx.layer_idx << " (slabs=" << ctx.prepared_store->expertSlabCount() << ")");
                    return true;
                }
            }

            if (gate_ref && up_ref && down_ref)
            {
                std::vector<int> missing_experts;
                missing_experts.reserve(required_experts.size());
                for (int e : required_experts)
                {
                    ctx.prepared_gate_gemm[e] = ctx.prepared_store->expertGemmKernel(*gate_ref, e);
                    ctx.prepared_up_gemm[e] = ctx.prepared_store->expertGemmKernel(*up_ref, e);
                    ctx.prepared_down_gemm[e] = ctx.prepared_store->expertGemmKernel(*down_ref, e);
                    if (!ctx.prepared_gate_gemm[e] ||
                        !ctx.prepared_up_gemm[e] ||
                        !ctx.prepared_down_gemm[e])
                    {
                        ctx.prepared_gate_gemm[e] = nullptr;
                        ctx.prepared_up_gemm[e] = nullptr;
                        ctx.prepared_down_gemm[e] = nullptr;
                        missing_experts.push_back(e);
                    }
                }

                if (!missing_experts.empty())
                {
                    ctx.gate_slab_ref = *gate_ref;
                    ctx.up_slab_ref = *up_ref;
                    ctx.down_slab_ref = *down_ref;
                    if (ctx.device_id.is_gpu())
                    {
                        LOG_ERROR("[MoEWeightService] PreparedWeightStore GPU expert slabs for layer "
                                  << ctx.layer_idx << " on " << ctx.device_id.to_string()
                                  << " are missing " << missing_experts.size()
                                  << " required expert(s); refusing duplicate GPU repack allocation");
                        return false;
                    }
                    experts_to_prep = std::move(missing_experts);
                    prep_count = static_cast<int>(experts_to_prep.size());
                    LOG_DEBUG("[MoEWeightService] PreparedWeightStore slabs for layer "
                              << ctx.layer_idx << " on " << ctx.device_id.to_string()
                              << " are missing " << prep_count
                              << " required expert(s); preparing incremental arrivals");
                }
            }
        }

        LOG_DEBUG("[MoEWeightService] Preparing GEMM engines for " << prep_count << "/" << num_experts
                                                                   << " experts (3 weights each = " << (prep_count * 3) << " total"
                                                                   << (use_mask ? " [dynamic rebalance: mask-active only]" : "") << ")...");

        // GPU path: unified H2D + GPU repack via LoadOrchestrator pipeline.
        if (ctx.device_id.is_gpu())
        {
            return prepareGemmEnginesGPU(ctx);
        }

        // CPU path: parallelize expert GEMM engine preparation.
        // Each expert has unique tensors (unique raw_data() keys), so no cache
        // key collisions.  The heavy VNNI interleave runs lock-free.
        // Phase D: prepareExpertGemmLocal returns shared_ptr without global registry.
        const int target_numa_node = ctx.cpu_numa_node;
        const bool enforce_numa_placement = target_numa_node >= 0;
        if (enforce_numa_placement && numa_available() < 0)
        {
            LOG_ERROR("[MoEWeightService][NUMA] Cannot enforce CPU expert packing on NUMA node "
                      << target_numa_node << ": libnuma policy APIs are unavailable");
            return false;
        }
        if (!enforce_numa_placement)
        {
            LOG_DEBUG("[MoEWeightService][NUMA] CPU expert packing uses aggregate CPU domain; "
                      "strict single-node placement is disabled for layer "
                      << ctx.layer_idx);
        }
        std::atomic<bool> error_flag{false};

        // Per-expert engine storage for parallel assignment (avoids push_back race)
        std::vector<std::shared_ptr<ITensorGemm>> local_gate_engines(prep_count);
        std::vector<std::shared_ptr<ITensorGemm>> local_up_engines(prep_count);
        std::vector<std::shared_ptr<ITensorGemm>> local_down_engines(prep_count);

#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < prep_count; ++idx)
        {
            if (error_flag.load(std::memory_order_relaxed))
                continue;
            const int e = experts_to_prep[idx];

            if (!ctx.expert_gate_views[e] || !ctx.expert_up_views[e] || !ctx.expert_down_views[e])
            {
                LOG_ERROR("[MoEWeightService] Null expert view for local expert " << e);
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }

            auto gate_engine = KernelFactory::prepareExpertGemmLocal(
                ctx.expert_gate_views[e].get(), ctx.device_id);
            auto up_engine = KernelFactory::prepareExpertGemmLocal(
                ctx.expert_up_views[e].get(), ctx.device_id);
            auto down_engine = KernelFactory::prepareExpertGemmLocal(
                ctx.expert_down_views[e].get(), ctx.device_id);

            if (!gate_engine || !up_engine || !down_engine)
            {
                LOG_ERROR("[MoEWeightService] Failed to prepare GEMM weights for expert " << e);
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }
            if (enforce_numa_placement &&
                (!enforceExpertKernelNUMA(gate_engine.get(), target_numa_node, ctx.layer_idx, e, "gate") ||
                 !enforceExpertKernelNUMA(up_engine.get(), target_numa_node, ctx.layer_idx, e, "up") ||
                 !enforceExpertKernelNUMA(down_engine.get(), target_numa_node, ctx.layer_idx, e, "down")))
            {
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }

            ctx.prepared_gate_gemm[e] = gate_engine.get();
            ctx.prepared_up_gemm[e] = up_engine.get();
            ctx.prepared_down_gemm[e] = down_engine.get();

            // Store in per-index slot (no contention — each idx is unique)
            local_gate_engines[idx] = std::move(gate_engine);
            local_up_engines[idx] = std::move(up_engine);
            local_down_engines[idx] = std::move(down_engine);
        }

        // Transfer engine lifetimes to ctx after parallel region (single-threaded)
        if (!error_flag.load())
        {
            ctx.moe_owned_kernels.reserve(ctx.moe_owned_kernels.size() + prep_count * 3);
            for (int idx = 0; idx < prep_count; ++idx)
            {
                if (local_gate_engines[idx])
                    ctx.moe_owned_kernels.push_back(std::move(local_gate_engines[idx]));
                if (local_up_engines[idx])
                    ctx.moe_owned_kernels.push_back(std::move(local_up_engines[idx]));
                if (local_down_engines[idx])
                    ctx.moe_owned_kernels.push_back(std::move(local_down_engines[idx]));
            }
        }

        if (error_flag.load())
        {
            return false;
        }

        LOG_DEBUG("[MoEWeightService] All " << (prep_count * 3) << " expert GEMM engines prepared (CPU/KernelFactory path)");

        // ── Phase B: Register store-owned expert engines ─────────────────────
        if (ctx.prepared_store)
        {
            bool store_owns_all_registered_engines = true;

            // Register three slabs (gate, up, down) for this layer.
            // Use the 3D parent tensor identity for source_identity when available.
            auto register_slab = [&](WeightRole role, const std::vector<ITensorGemm *> &engines,
                                     const std::vector<std::shared_ptr<TensorBase>> &views,
                                     TensorBase *parent_3d,
                                     const ExpertSlabRef *existing_ref) -> ExpertSlabRef
            {
                ExpertSlabDescriptor desc = makeExpertSlabDescriptor(ctx, role);
                // source_identity left default — TensorBase doesn't expose WeightIdentity.
                // Phase C will wire this through WeightManager lookup.
                (void)parent_3d;

                auto slab_ref = existing_ref ? *existing_ref : ctx.prepared_store->registerExpertSlab(desc);

                // Build arrivals for all prepared experts
                std::vector<ExpertArrival> arrivals;
                arrivals.reserve(experts_to_prep.size());
                for (int e : experts_to_prep)
                {
                    if (!engines[e])
                        continue;
                    ExpertArrival arrival;
                    arrival.expert_id = e;
                    arrival.engine = engines[e];
                    // Find engine_lifetime from moe_owned_kernels
                    for (const auto &owned : ctx.moe_owned_kernels)
                    {
                        if (owned.get() == engines[e])
                        {
                            arrival.engine_lifetime = owned;
                            break;
                        }
                    }
                    if (!arrival.engine_lifetime)
                    {
                        store_owns_all_registered_engines = false;
                        LOG_WARN("[MoEWeightService] PreparedWeightStore registration for layer "
                                 << ctx.layer_idx << " expert " << e
                                 << " role " << static_cast<int>(role)
                                 << " has no shared engine lifetime; retaining graph-local owner");
                    }
                    arrival.view_lifetime = (e < static_cast<int>(views.size())) ? views[e] : nullptr;
                    arrival.derivation = WeightDerivationKind::ExpertSlice;
                    arrivals.push_back(std::move(arrival));
                }

                ctx.prepared_store->registerArrivedExperts(slab_ref, arrivals);
                return slab_ref;
            };

            ctx.gate_slab_ref = register_slab(WeightRole::MoEExpertGate, ctx.prepared_gate_gemm,
                                              ctx.expert_gate_views, ctx.gate_exps,
                                              ctx.gate_slab_ref ? &*ctx.gate_slab_ref : nullptr);
            ctx.up_slab_ref = register_slab(WeightRole::MoEExpertUp, ctx.prepared_up_gemm,
                                            ctx.expert_up_views, ctx.up_exps,
                                            ctx.up_slab_ref ? &*ctx.up_slab_ref : nullptr);
            ctx.down_slab_ref = register_slab(WeightRole::MoEExpertDown, ctx.prepared_down_gemm,
                                              ctx.expert_down_views, ctx.down_exps,
                                              ctx.down_slab_ref ? &*ctx.down_slab_ref : nullptr);

            LOG_DEBUG("[MoEWeightService] Phase B: Registered " << (prep_count * 3)
                                                                << " expert engines in PreparedWeightStore ("
                                                                << ctx.prepared_store->expertSlabCount() << " slabs total, "
                                                                << ctx.prepared_store->totalPopulatedExperts() << " populated experts)");

            if (store_owns_all_registered_engines)
            {
                ctx.moe_owned_kernels.clear();
                ctx.moe_owned_kernels.shrink_to_fit();
                LOG_DEBUG("[MoEWeightService] Released graph-local shared ownership for store-backed CPU expert engines");
            }
        }

        // NUMA audit: verify packed weights landed on the requested NUMA node.
        // Aggregate CPU domains intentionally skip this single-node check.
        if (enforce_numa_placement)
        {
            auditExpertNUMA(experts_to_prep, ctx.prepared_gate_gemm,
                            "initial_pack", ctx.layer_idx, target_numa_node);
        }

        // Mark experts as prepared in the payload provider (enables host data release)
        if (ctx.payload_provider)
        {
            for (int e : experts_to_prep)
                ctx.payload_provider->markExpertPrepared(ctx.layer_idx, e);
        }

        // Release mmap pages backing the raw expert weight data.
        // The VNNI interleaved engines now own their own copy — the original
        // mmap data is never accessed again. Releasing per-layer reduces peak RSS
        // by ~500 MB/layer instead of waiting for a bulk release at the end.
        // NOTE: Only safe for mmap-backed tensors. Expert-ID-sliced tensors
        // are heap-allocated copies — MADV_DONTNEED on heap memory corrupts malloc metadata.
        if (ctx.advise_raw_pages_after_prepare)
        {
            size_t released = 0;
            if (ctx.gate_exps && ctx.gate_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(ctx.gate_exps->raw_data(), ctx.gate_exps->size_bytes());
            if (ctx.up_exps && ctx.up_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(ctx.up_exps->raw_data(), ctx.up_exps->size_bytes());
            if (ctx.down_exps && ctx.down_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(ctx.down_exps->raw_data(), ctx.down_exps->size_bytes());
            if (released > 0)
                LOG_DEBUG("[MoEWeightService] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after engine packing");
        }

        return true;
    }

    // =========================================================================
    // releaseRawWeights — Free 3D parent weight tensors
    // =========================================================================

    size_t MoEExpertWeightService::releaseRawWeights(MoEWeightContext &ctx)
    {
        size_t freed = 0;

        auto release_3d = [&](TensorBase *&tensor, const char *name)
        {
            if (!tensor)
                return;
            if (tensor->is_mmap_data())
            {
                freed += tensor->size_bytes();
                LOG_DEBUG("[MoEWeightService] " << name << ": mmap-backed ("
                                                << (tensor->size_bytes() >> 20) << " MB) — already DONTNEED");
            }
            else if (!tensor->is_raw_data_released())
            {
                size_t bytes = tensor->size_bytes();
                tensor->release_raw_data();
                freed += bytes;
                LOG_DEBUG("[MoEWeightService] " << name << ": released "
                                                << (bytes >> 20) << " MB heap data");
            }
        };

        release_3d(ctx.gate_exps, "gate_exps");
        release_3d(ctx.up_exps, "up_exps");
        release_3d(ctx.down_exps, "down_exps");

        // Null out 3D pointers to prevent accidental fallback to raw repacking.
        ctx.gate_exps = nullptr;
        ctx.up_exps = nullptr;
        ctx.down_exps = nullptr;

        if (freed > 0)
        {
            LOG_DEBUG("[MoEWeightService] Layer " << ctx.layer_idx
                                                 << ": released " << (freed >> 20) << " MB raw expert weights"
                                                 << " (VNNI engines retain packed data)");
        }

        return freed;
    }

    // =========================================================================
    // Weight serialization (for MPI transfer)
    // =========================================================================

    ExpertWeightBlobs MoEExpertWeightService::detachAndSerializeExpert(MoEWeightContext &ctx, int expert_id)
    {
        ExpertWeightBlobs blobs;

        auto serialize_proj = [&](ITensorGemm *engine, const char * /*proj_name*/) -> std::vector<uint8_t>
        {
            if (!engine)
                return {};
            if (!engine->hasWeights())
                return {};
            auto packed = engine->detachWeights();
            if (!packed)
                return {};
            return packed_weights_serialization::serialize(*packed);
        };

        blobs.gate = serialize_proj(ctx.prepared_gate_gemm[expert_id], "gate");
        blobs.up = serialize_proj(ctx.prepared_up_gemm[expert_id], "up");
        blobs.down = serialize_proj(ctx.prepared_down_gemm[expert_id], "down");

        return blobs;
    }

    ExpertWeightBlobs MoEExpertWeightService::serializeExpert(const MoEWeightContext &ctx, int expert_id)
    {
        using namespace cpu::native_vnni;

        ExpertWeightBlobs blobs;

        auto serialize_proj = [expert_id, &ctx](ITensorGemm *engine, const char *proj_name) -> std::vector<uint8_t>
        {
            if (!engine || !engine->hasWeights())
            {
                LOG_ERROR("[MoEWeightService] Cannot serialize packed expert weight: layer "
                          << ctx.layer_idx << " expert " << expert_id << " " << proj_name
                          << " has no prepared GEMM engine");
                return {};
            }

            auto packed = engine->cloneWeights();
            if (!packed)
            {
                LOG_ERROR("[MoEWeightService] Cannot serialize packed expert weight: layer "
                          << ctx.layer_idx << " expert " << expert_id << " " << proj_name
                          << " engine cannot clone packed weights");
                return {};
            }

            return packed_weights_serialization::serialize(*packed);
        };

        blobs.gate = serialize_proj(ctx.prepared_gate_gemm[expert_id], "gate");
        blobs.up = serialize_proj(ctx.prepared_up_gemm[expert_id], "up");
        blobs.down = serialize_proj(ctx.prepared_down_gemm[expert_id], "down");

        return blobs;
    }

    // =========================================================================
    // Phased rebalance API
    // =========================================================================

    std::vector<const TensorBase *> MoEExpertWeightService::releaseDepartedExperts(
        MoEWeightContext &ctx, const std::vector<bool> &new_mask)
    {
        if (new_mask.size() < static_cast<size_t>(std::max(0, ctx.num_experts)))
        {
            throw std::runtime_error(
                "MoEExpertWeightService::releaseDepartedExperts received a mask smaller than num_experts");
        }

        std::vector<const TensorBase *> evict_tensors;
        std::vector<int> departed_ids;
        std::vector<ITensorGemm *> departed_engines;

        auto remember_departed_engine = [&](ITensorGemm *engine)
        {
            if (!engine)
                return;
            if (std::find(departed_engines.begin(), departed_engines.end(), engine) == departed_engines.end())
                departed_engines.push_back(engine);
        };

        for (int e = 0; e < ctx.num_experts; ++e)
        {
            const bool has_any_projection =
                ctx.prepared_gate_gemm[e] ||
                ctx.prepared_up_gemm[e] ||
                ctx.prepared_down_gemm[e];
            if (!new_mask[e] && has_any_projection)
            {
                departed_ids.push_back(e);
                remember_departed_engine(ctx.prepared_gate_gemm[e]);
                remember_departed_engine(ctx.prepared_up_gemm[e]);
                remember_departed_engine(ctx.prepared_down_gemm[e]);

                // Collect tensor views for batch cache eviction by the caller
                if (e < static_cast<int>(ctx.expert_gate_views.size()) && ctx.expert_gate_views[e])
                    evict_tensors.push_back(ctx.expert_gate_views[e].get());
                if (e < static_cast<int>(ctx.expert_up_views.size()) && ctx.expert_up_views[e])
                    evict_tensors.push_back(ctx.expert_up_views[e].get());
                if (e < static_cast<int>(ctx.expert_down_views.size()) && ctx.expert_down_views[e])
                    evict_tensors.push_back(ctx.expert_down_views[e].get());

                // Null out the engine pointers
                ctx.prepared_gate_gemm[e] = nullptr;
                ctx.prepared_up_gemm[e] = nullptr;
                ctx.prepared_down_gemm[e] = nullptr;
            }
        }

        // Mirror departures to ExpertGemmRegistry (keeps registry in sync with stage state)
        if (ctx.expert_registry && !departed_ids.empty())
        {
            for (int e : departed_ids)
            {
                ctx.expert_registry->removeEngine(ctx.device_id, ctx.layer_idx, e, ExpertGemmRegistry::WeightRole::GATE);
                ctx.expert_registry->removeEngine(ctx.device_id, ctx.layer_idx, e, ExpertGemmRegistry::WeightRole::UP);
                ctx.expert_registry->removeEngine(ctx.device_id, ctx.layer_idx, e, ExpertGemmRegistry::WeightRole::DOWN);
            }
            LOG_DEBUG("[MoEWeightService] Removed " << departed_ids.size()
                                                    << " departed experts from ExpertGemmRegistry (layer " << ctx.layer_idx << ")");
        }

        // Phase C: Notify PreparedWeightStore of departed experts
        if (ctx.prepared_store && !departed_ids.empty())
        {
            if (ctx.gate_slab_ref.has_value())
                ctx.prepared_store->releaseDepartedExperts(*ctx.gate_slab_ref, departed_ids);
            if (ctx.up_slab_ref.has_value())
                ctx.prepared_store->releaseDepartedExperts(*ctx.up_slab_ref, departed_ids);
            if (ctx.down_slab_ref.has_value())
                ctx.prepared_store->releaseDepartedExperts(*ctx.down_slab_ref, departed_ids);

            LOG_DEBUG("[MoEWeightService] Phase C: Released " << departed_ids.size()
                                                              << " departed experts from PreparedWeightStore");
        }

        if (!departed_engines.empty() && !ctx.moe_owned_kernels.empty())
        {
            std::vector<ITensorGemm *> released_owned_engines;
            for (const auto &owned : ctx.moe_owned_kernels)
            {
                if (!owned)
                    continue;
                ITensorGemm *raw = owned.get();
                if (std::find(departed_engines.begin(), departed_engines.end(), raw) == departed_engines.end())
                    continue;
                if (std::find(released_owned_engines.begin(), released_owned_engines.end(), raw) != released_owned_engines.end())
                    continue;
                owned->releaseWeights();
                released_owned_engines.push_back(raw);
            }

            const size_t before = ctx.moe_owned_kernels.size();
            ctx.moe_owned_kernels.erase(
                std::remove_if(ctx.moe_owned_kernels.begin(),
                               ctx.moe_owned_kernels.end(),
                               [&](const std::shared_ptr<ITensorGemm> &owned)
                               {
                                   if (!owned)
                                       return false;
                                   return std::find(departed_engines.begin(),
                                                    departed_engines.end(),
                                                    owned.get()) != departed_engines.end();
                               }),
                ctx.moe_owned_kernels.end());
            const size_t released = before - ctx.moe_owned_kernels.size();
            if (released > 0)
            {
                LOG_DEBUG("[MoEWeightService] Released " << released
                                                         << " graph-local expert GEMM owners for "
                                                         << departed_ids.size()
                                                         << " departed experts on layer "
                                                         << ctx.layer_idx);
            }
        }

        return evict_tensors;
    }

    bool MoEExpertWeightService::registerAndPrepareNewExperts(
        MoEWeightContext &ctx,
        const std::vector<bool> &new_mask,
        const std::unordered_map<int, ExpertWeightBlobs> *received_weights)
    {
        // Find newly-acquired experts (true in new_mask, not previously prepared)
        std::vector<int> new_experts;
        for (int e = 0; e < ctx.num_experts; ++e)
        {
            const bool has_complete_engine =
                ctx.prepared_gate_gemm[e] &&
                ctx.prepared_up_gemm[e] &&
                ctx.prepared_down_gemm[e];
            if (new_mask[e] && !has_complete_engine)
                new_experts.push_back(e);
        }
        if (new_experts.empty())
            return true;

        // GPU path: prefer transferred packed/source-domain payloads, converting
        // CPU-native-with-blocks to GPU-repacked layout via the device pipeline.
        // Raw GGUF data is only a fallback for initial materialization before host
        // source bytes are released.
        if (ctx.device_id.is_gpu())
        {
            return registerAndPrepareNewExpertsGPU(ctx, new_experts, received_weights);
        }

        // CPU path: deserialize transferred weights or resolve existing store-owned engines.
        auto t_start = std::chrono::high_resolution_clock::now();
        int transferred_count = 0;
        std::atomic<bool> error_flag{false};
        const int count = static_cast<int>(new_experts.size());
        const int target_numa_node = ctx.cpu_numa_node;
        const bool enforce_numa_placement = target_numa_node >= 0;
        if (enforce_numa_placement && numa_available() < 0)
        {
            LOG_ERROR("[MoEWeightService][NUMA] Cannot enforce CPU expert arrivals on NUMA node "
                      << target_numa_node << ": libnuma policy APIs are unavailable");
            return false;
        }
        if (!enforce_numa_placement)
        {
            LOG_DEBUG("[MoEWeightService][NUMA] CPU expert arrivals use aggregate CPU domain; "
                      "strict single-node placement is disabled for layer "
                      << ctx.layer_idx);
        }

        auto cached_engine_for = [&](const std::optional<ExpertSlabRef> &slab_ref, int expert_id) -> ITensorGemm *
        {
            if (!ctx.prepared_store || !slab_ref.has_value())
                return nullptr;
            return ctx.prepared_store->expertGemmKernel(*slab_ref, expert_id);
        };

        // Prepare engines: use store-owned engines first, then transferred blobs.
        // Raw tensor repacking is intentionally forbidden after initial eager graph
        // materialization because host expert data may already have been released.
        for (int idx = 0; idx < count; ++idx)
        {
            if (error_flag.load(std::memory_order_relaxed))
                break;
            const int e = new_experts[idx];

            std::shared_ptr<ITensorGemm> gate_engine, up_engine, down_engine;

            ITensorGemm *cached_gate = cached_engine_for(ctx.gate_slab_ref, e);
            ITensorGemm *cached_up = cached_engine_for(ctx.up_slab_ref, e);
            ITensorGemm *cached_down = cached_engine_for(ctx.down_slab_ref, e);
            if (cached_gate && cached_up && cached_down)
            {
                ctx.prepared_gate_gemm[e] = cached_gate;
                ctx.prepared_up_gemm[e] = cached_up;
                ctx.prepared_down_gemm[e] = cached_down;
                continue;
            }

            // Fast path: create kernels directly from pre-packed transferred blobs
            const ExpertWeightBlobs *blobs = nullptr;
            if (received_weights)
            {
                auto it = received_weights->find(e);
                if (it != received_weights->end() && !it->second.empty())
                    blobs = &it->second;
            }

            if (!blobs)
            {
                LOG_ERROR("[MoEWeightService] Missing transferred/store-owned packed weights for new CPU expert "
                          << e << " on layer " << ctx.layer_idx << "; raw expert repack fallback is disabled");
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }

            gate_engine = KernelFactory::createExpertGemmFromTransferBlob(blobs->gate);
            up_engine = KernelFactory::createExpertGemmFromTransferBlob(blobs->up);
            down_engine = KernelFactory::createExpertGemmFromTransferBlob(blobs->down);

            if (gate_engine && up_engine && down_engine)
                ++transferred_count;

            if (!gate_engine || !up_engine || !down_engine)
            {
                LOG_ERROR("[MoEWeightService] Failed to deserialize transferred packed GEMM weights for new CPU expert "
                          << e << " on layer " << ctx.layer_idx << "; raw expert repack fallback is disabled");
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }
            if (enforce_numa_placement &&
                (!enforceExpertKernelNUMA(gate_engine.get(), target_numa_node, ctx.layer_idx, e, "gate") ||
                 !enforceExpertKernelNUMA(up_engine.get(), target_numa_node, ctx.layer_idx, e, "up") ||
                 !enforceExpertKernelNUMA(down_engine.get(), target_numa_node, ctx.layer_idx, e, "down")))
            {
                error_flag.store(true, std::memory_order_relaxed);
                continue;
            }

            ctx.prepared_gate_gemm[e] = gate_engine.get();
            ctx.prepared_up_gemm[e] = up_engine.get();
            ctx.prepared_down_gemm[e] = down_engine.get();

            ctx.moe_owned_kernels.push_back(std::move(gate_engine));
            ctx.moe_owned_kernels.push_back(std::move(up_engine));
            ctx.moe_owned_kernels.push_back(std::move(down_engine));
        }

        if (error_flag.load())
            return false;

        // Mark experts as prepared in the payload provider
        if (ctx.payload_provider)
        {
            for (int e : new_experts)
                ctx.payload_provider->markExpertPrepared(ctx.layer_idx, e);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double prep_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        LOG_DEBUG("[MoEWeightService] Engine prep for " << new_experts.size()
                                                        << " experts (" << transferred_count << " transferred): "
                                                        << std::fixed << std::setprecision(1) << prep_ms << " ms");

        if (enforce_numa_placement)
        {
            auditExpertNUMA(new_experts, ctx.prepared_gate_gemm,
                            (transferred_count > 0 ? "rebalance_transferred" : "rebalance_repacked"),
                            ctx.layer_idx, target_numa_node);
        }

        // Phase C: Register new arrivals in PreparedWeightStore using cached slab refs
        if (ctx.prepared_store && !new_experts.empty())
        {
            auto register_rebalance_arrivals = [&](const std::optional<ExpertSlabRef> &slab_ref,
                                                   const std::vector<ITensorGemm *> &engines,
                                                   const std::vector<std::shared_ptr<TensorBase>> &views)
            {
                if (!slab_ref.has_value())
                    return;

                std::vector<ExpertArrival> arrivals;
                for (int e : new_experts)
                {
                    if (e < 0 || e >= static_cast<int>(engines.size()) || !engines[e])
                        continue;
                    ExpertArrival arrival;
                    arrival.expert_id = e;
                    arrival.engine = engines[e];
                    // Find engine_lifetime from moe_owned_kernels
                    for (const auto &owned : ctx.moe_owned_kernels)
                    {
                        if (owned.get() == engines[e])
                        {
                            arrival.engine_lifetime = owned;
                            break;
                        }
                    }
                    arrival.view_lifetime = (e < static_cast<int>(views.size())) ? views[e] : nullptr;
                    arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
                    arrivals.push_back(std::move(arrival));
                }

                if (!arrivals.empty())
                    ctx.prepared_store->registerArrivedExperts(*slab_ref, arrivals);
            };

            register_rebalance_arrivals(ctx.gate_slab_ref, ctx.prepared_gate_gemm, ctx.expert_gate_views);
            register_rebalance_arrivals(ctx.up_slab_ref, ctx.prepared_up_gemm, ctx.expert_up_views);
            register_rebalance_arrivals(ctx.down_slab_ref, ctx.prepared_down_gemm, ctx.expert_down_views);

            LOG_DEBUG("[MoEWeightService] Phase C rebalance: Registered "
                      << new_experts.size() * 3 << " new expert engines in PreparedWeightStore"
                      << " (using cached slab refs)");
        }

        return true;
    }

    // =========================================================================
    // GPU rebalance path (LoadOrchestrator: raw H2D + GPU repack for new experts)
    // =========================================================================

    bool MoEExpertWeightService::registerAndPrepareNewExpertsGPU(
        MoEWeightContext &ctx,
        const std::vector<int> &new_experts,
        const std::unordered_map<int, ExpertWeightBlobs> *received_weights)
    {
        using Clock = std::chrono::high_resolution_clock;
        const auto t_start = Clock::now();

        const int gpu_ordinal = ctx.device_id.is_cuda()
                                    ? ctx.device_id.cuda_ordinal()
                                    : ctx.device_id.rocm_ordinal();
        IBackend *backend = getBackendFor(ctx.device_id);
        if (!backend)
        {
            LOG_ERROR("[MoEWeightService::GPU-rebalance] No backend for "
                      << ctx.device_id.to_string());
            return false;
        }

        // Weight groups: gate, up, down
        struct WeightGroup
        {
            const char *label;
            std::vector<std::shared_ptr<TensorBase>> &views;
            std::vector<ITensorGemm *> &out_gemms;
        };
        WeightGroup groups[] = {
            {"gate", ctx.expert_gate_views, ctx.prepared_gate_gemm},
            {"up", ctx.expert_up_views, ctx.prepared_up_gemm},
            {"down", ctx.expert_down_views, ctx.prepared_down_gemm},
        };
        resolveExpertSlabRefs(ctx, /*create_if_missing=*/false);

        auto orchestrator = std::make_shared<LoadOrchestrator>(backend);
        orchestrator->addDevice(gpu_ordinal);

        size_t max_raw_bytes = 0;
        size_t total_planned = 0;

        struct TransferSource
        {
            std::unique_ptr<IPackedWeights> packed;
            const cpu::native_vnni::CPUPackedWeightsWithNativeBlocks *native = nullptr;
            RepackFormat format = RepackFormat::Q4_0;
        };
        std::vector<TransferSource> transfer_sources;
        std::unordered_map<int, ExpertWeightBlobs> provider_payload_cache;

        auto blobsForExpert = [&](int expert_id) -> const ExpertWeightBlobs *
        {
            if (received_weights)
            {
                auto it = received_weights->find(expert_id);
                if (it != received_weights->end() && !it->second.empty())
                    return &it->second;
            }

            auto cached = provider_payload_cache.find(expert_id);
            if (cached != provider_payload_cache.end())
                return &cached->second;

            if (ctx.payload_provider)
            {
                auto payload = ctx.payload_provider->payloadFor(ctx.layer_idx, expert_id);
                if (payload && !payload->empty())
                {
                    auto [it, inserted] = provider_payload_cache.emplace(expert_id, std::move(*payload));
                    (void)inserted;
                    return &it->second;
                }
            }

            return nullptr;
        };

        auto blobFor = [&](int expert_id, const char *label) -> const std::vector<uint8_t> *
        {
            const ExpertWeightBlobs *blobs = blobsForExpert(expert_id);
            if (!blobs)
                return nullptr;
            if (std::strcmp(label, "gate") == 0)
                return &blobs->gate;
            if (std::strcmp(label, "up") == 0)
                return &blobs->up;
            return &blobs->down;
        };

        auto hasCompleteTransferredBlobs = [&](int expert_id)
        {
            for (auto &grp : groups)
            {
                auto *blob = blobFor(expert_id, grp.label);
                if (!blob || blob->empty())
                    return false;
            }
            return true;
        };

        auto makeTransferSource = [&](const std::vector<uint8_t> &blob,
                                      int expert_id,
                                      const char *label) -> std::optional<TransferSource>
        {
            if (blob.empty())
                return std::nullopt;
            auto packed = packed_weights_serialization::deserialize(blob.data(), blob.size());
            if (!packed)
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Failed to deserialize transferred packed weights for expert "
                          << expert_id << " " << label << " on " << ctx.device_id.to_string());
                return std::nullopt;
            }

            auto *native = dynamic_cast<cpu::native_vnni::CPUPackedWeightsWithNativeBlocks *>(packed.get());
            if (!native)
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Transferred expert " << expert_id << " " << label
                                                                                  << " lacks native quantized blocks; CPU-packed-only conversion to GPU is not available yet");
                return std::nullopt;
            }

            const auto &cpu_packed = native->packed();
            auto repack_fmt = codebookIdToRepackFormat(cpu_packed.codebook_id, cpu_packed.is_superblock);
            if (!repack_fmt)
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Unsupported transferred packed format for expert "
                          << expert_id << " " << label
                          << " (codebook=" << static_cast<int>(cpu_packed.codebook_id)
                          << ", superblock=" << cpu_packed.is_superblock << ")");
                return std::nullopt;
            }

            TransferSource source;
            source.native = native;
            source.format = *repack_fmt;
            source.packed = std::move(packed);
            return source;
        };

        std::vector<int> experts_to_load;
        experts_to_load.reserve(new_experts.size());
        auto cached_engine_for = [&](std::optional<ExpertSlabRef> &slab_ref,
                                     WeightRole role,
                                     int expert_id) -> ITensorGemm *
        {
            if (!ctx.prepared_store || expert_id < 0 || expert_id >= ctx.num_experts)
                return nullptr;
            if (slab_ref.has_value())
            {
                if (auto *engine = ctx.prepared_store->expertGemmKernel(*slab_ref, expert_id))
                    return engine;
            }
            auto found = ctx.prepared_store->findExpertSlab(makeExpertSlabDescriptor(ctx, role));
            if (!found.has_value())
                return nullptr;
            slab_ref = *found;
            return ctx.prepared_store->expertGemmKernel(*found, expert_id);
        };
        for (int e : new_experts)
        {
            ITensorGemm *cached_gate = cached_engine_for(ctx.gate_slab_ref, WeightRole::MoEExpertGate, e);
            ITensorGemm *cached_up = cached_engine_for(ctx.up_slab_ref, WeightRole::MoEExpertUp, e);
            ITensorGemm *cached_down = cached_engine_for(ctx.down_slab_ref, WeightRole::MoEExpertDown, e);
            if (cached_gate && cached_up && cached_down)
            {
                groups[0].out_gemms[e] = cached_gate;
                groups[1].out_gemms[e] = cached_up;
                groups[2].out_gemms[e] = cached_down;
                continue;
            }

            if (!hasCompleteTransferredBlobs(e))
            {
                LOG_ERROR("[MoEWeightService::GPU-rebalance] Expert " << e
                                                                      << " requires transferred/provider blobs for all gate/up/down weights on "
                                                                      << ctx.device_id.to_string() << " (layer " << ctx.layer_idx
                                                                      << "). Raw GGUF fallback is not allowed during GPU rebalance after host release.");
                return false;
            }

            experts_to_load.push_back(e);
        }

        const int count = static_cast<int>(experts_to_load.size());
        if (count == 0)
        {
            LOG_DEBUG("[MoEWeightService::GPU-rebalance] All " << new_experts.size()
                                                               << " requested experts already had prepared GPU handles on "
                                                               << ctx.device_id.to_string() << " (layer " << ctx.layer_idx << ")");
            return true;
        }

        // Phase 1: Plan weights for new experts
        for (auto &grp : groups)
        {
            for (int idx = 0; idx < count; ++idx)
            {
                const int e = experts_to_load[idx];
                const auto &view = grp.views[e];
                if (!view)
                {
                    LOG_ERROR("[MoEWeightService::GPU-rebalance] Null view for expert "
                              << e << " in " << grp.label);
                    return false;
                }

                auto *unpackable = dynamic_cast<IINT8Unpackable *>(view.get());
                const NativeVnniFormatInfo *vnni = unpackable ? unpackable->vnniFormatInfo() : nullptr;
                if (!vnni)
                {
                    LOG_ERROR("[MoEWeightService::GPU-rebalance] Expert " << e << " "
                                                                          << grp.label << " has no VNNI format info");
                    return false;
                }

#ifdef HAVE_ROCM
                if (ctx.device_id.is_rocm() && vnni->codebook_id >= 11 && vnni->codebook_id <= 17)
                {
                    if (!rocm::ensureIQGridTablesInitialized(gpu_ordinal))
                    {
                        LOG_ERROR("[MoEWeightService::GPU-rebalance] Failed to initialize ROCm IQ grid tables for "
                                  << ctx.device_id.to_string());
                        return false;
                    }
                }
#endif

                const int N = static_cast<int>(view->rows());
                const int K = static_cast<int>(view->cols());
                const size_t raw_bytes = quantizedViewRawBytes(*view);
                if (raw_bytes == 0)
                {
                    LOG_ERROR("[MoEWeightService::GPU-rebalance] Could not determine raw byte size for expert "
                              << e << " " << grp.label);
                    return false;
                }
                const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

                auto *blob = blobFor(e, grp.label);
                if (!blob || blob->empty())
                {
                    LOG_ERROR("[MoEWeightService::GPU-rebalance] Missing required transferred/provider blob for expert "
                              << e << " " << grp.label << " on " << ctx.device_id.to_string()
                              << " (layer " << ctx.layer_idx << ")");
                    return false;
                }

                auto source = makeTransferSource(*blob, e, grp.label);
                if (!source)
                    return false;
                const size_t source_bytes = source->native->nativeBlocks().size();
                transfer_sources.push_back(std::move(*source));

                orchestrator->planWeight(gpu_ordinal, slot_name, N, K,
                                         vnni->payload_bytes, vnni->is_asymmetric,
                                         vnni->has_emins, source_bytes);
                max_raw_bytes = std::max(max_raw_bytes, source_bytes);
                ++total_planned;
            }
        }

        LOG_DEBUG("[MoEWeightService::GPU-rebalance] Planned " << total_planned
                                                               << " expert weights for " << ctx.device_id.to_string()
                                                               << " (layer " << ctx.layer_idx << ")");

        // Phase 2: Allocate VRAM pool + pinned ring buffer
        const auto &rocm_cfg = debugEnv().rocm;
        const int repack_streams = std::clamp(rocm_cfg.repack_streams, 1, 8);
        const size_t staging_budget_bytes = rocm_cfg.repack_budget_mb > 0
                                                ? static_cast<size_t>(rocm_cfg.repack_budget_mb) * 1024ULL * 1024ULL
                                                : 0;
        const size_t staging_slot_bytes = staging_budget_bytes > 0
                                              ? std::min(max_raw_bytes,
                                                         std::max<size_t>(1, staging_budget_bytes /
                                                                                 static_cast<size_t>(repack_streams)))
                                              : max_raw_bytes;
        orchestrator->allocate(staging_slot_bytes, repack_streams);

        // Phase 3: Create weight jobs from raw GGUF data
        size_t transfer_source_idx = 0;
        for (auto &grp : groups)
        {
            for (int idx = 0; idx < count; ++idx)
            {
                const int e = experts_to_load[idx];
                const auto &view = grp.views[e];

                auto *unpackable = dynamic_cast<IINT8Unpackable *>(view.get());
                const NativeVnniFormatInfo *vnni = unpackable->vnniFormatInfo();

                auto repack_fmt = codebookIdToRepackFormat(vnni->codebook_id, vnni->is_superblock);
                if (!repack_fmt)
                {
                    LOG_ERROR("[MoEWeightService::GPU-rebalance] Unsupported repack format for expert "
                              << e << " " << grp.label
                              << " (codebook=" << static_cast<int>(vnni->codebook_id)
                              << ", superblock=" << vnni->is_superblock << ")");
                    return false;
                }

                const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

                WeightJob job;
                job.name = slot_name;
                job.format = *repack_fmt;
                job.N = static_cast<int>(view->rows());
                job.K = static_cast<int>(view->cols());
                job.is_asymmetric = vnni->is_asymmetric;

                if (transfer_source_idx >= transfer_sources.size())
                {
                    LOG_ERROR("[MoEWeightService::GPU-rebalance] Internal transfer source accounting mismatch");
                    return false;
                }
                const auto &source = transfer_sources[transfer_source_idx++];
                job.host_raw_data = source.native->nativeBlocks().data();
                job.raw_bytes = source.native->nativeBlocks().size();
                job.format = source.format;

                orchestrator->addWeightJob(gpu_ordinal, job);
            }
        }

        // Phase 4: Execute pipeline (pipelined H2D + GPU repack)
        orchestrator->load();

        auto *pool = orchestrator->getPool(gpu_ordinal);
        if (!pool)
        {
            LOG_ERROR("[MoEWeightService::GPU-rebalance] Pool not found after load");
            return false;
        }

        // Phase 5: Create per-expert GEMM kernels from pool slots
        for (auto &grp : groups)
        {
            for (int idx = 0; idx < count; ++idx)
            {
                const int e = experts_to_load[idx];
                const auto &view = grp.views[e];
                const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

                auto slot = pool->getSlot(slot_name);
                if (!slot)
                {
                    LOG_ERROR("[MoEWeightService::GPU-rebalance] No slot for " << slot_name);
                    return false;
                }

                auto *unpackable = dynamic_cast<IINT8Unpackable *>(view.get());
                const NativeVnniFormatInfo *vnni = unpackable->vnniFormatInfo();
                const int N = static_cast<int>(view->rows());
                const int K = static_cast<int>(view->cols());
                const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);

                std::shared_ptr<ITensorGemm> kernel;

#ifdef HAVE_CUDA
                if (ctx.device_id.is_cuda())
                {
                    kernel = std::make_shared<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                        N, K, gpu_ordinal,
                        slot->d_native_vnni_payload,
                        static_cast<uint16_t *>(slot->d_native_vnni_scales),
                        static_cast<uint16_t *>(slot->d_native_vnni_mins),
                        static_cast<uint32_t *>(slot->d_native_vnni_emins),
                        vnni->codebook_id, blocks_per_row,
                        orchestrator);
                }
#endif
#ifdef HAVE_ROCM
                if (ctx.device_id.is_rocm())
                {
                    kernel = std::make_shared<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                        N, K, gpu_ordinal,
                        slot->d_native_vnni_payload,
                        slot->d_native_vnni_scales,
                        slot->d_native_vnni_mins,
                        slot->d_native_vnni_emins,
                        vnni->codebook_id, blocks_per_row,
                        orchestrator);
                }
#endif

                if (!kernel)
                {
                    LOG_ERROR("[MoEWeightService::GPU-rebalance] Failed to create kernel for " << slot_name);
                    return false;
                }

                grp.out_gemms[e] = kernel.get();
                ctx.moe_owned_kernels.push_back(std::move(kernel));
            }
        }

        // Phase 6: Release staging resources; VRAM pool stays alive via kernel lifetime_owner_
        orchestrator->finalize();

        // Phase 7: Mark experts as prepared in the payload provider
        if (ctx.payload_provider)
        {
            for (int e : experts_to_load)
                ctx.payload_provider->markExpertPrepared(ctx.layer_idx, e);
        }

        // Mirror arrivals to ExpertGemmRegistry (keeps registry in sync with stage state)
        if (ctx.expert_registry)
        {
            for (int e : experts_to_load)
            {
                auto find_ownership = [&](ITensorGemm *raw) -> std::shared_ptr<ITensorGemm>
                {
                    for (const auto &owned : ctx.moe_owned_kernels)
                        if (owned.get() == raw)
                            return owned;
                    return nullptr;
                };

                if (ctx.prepared_gate_gemm[e])
                    ctx.expert_registry->replaceEngine(ctx.device_id, ctx.layer_idx, e,
                                                       ExpertGemmRegistry::WeightRole::GATE,
                                                       ctx.prepared_gate_gemm[e], find_ownership(ctx.prepared_gate_gemm[e]));
                if (ctx.prepared_up_gemm[e])
                    ctx.expert_registry->replaceEngine(ctx.device_id, ctx.layer_idx, e,
                                                       ExpertGemmRegistry::WeightRole::UP,
                                                       ctx.prepared_up_gemm[e], find_ownership(ctx.prepared_up_gemm[e]));
                if (ctx.prepared_down_gemm[e])
                    ctx.expert_registry->replaceEngine(ctx.device_id, ctx.layer_idx, e,
                                                       ExpertGemmRegistry::WeightRole::DOWN,
                                                       ctx.prepared_down_gemm[e], find_ownership(ctx.prepared_down_gemm[e]));
            }
            LOG_DEBUG("[MoEWeightService::GPU-rebalance] Registered " << experts_to_load.size()
                                                                      << " arrived experts in ExpertGemmRegistry (layer " << ctx.layer_idx << ")");
        }

        const auto elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        LOG_DEBUG("[MoEWeightService::GPU-rebalance] " << (count * 3) << " expert GEMM engines "
                                                      << "prepared via GPU pipeline in "
                                                      << std::fixed << std::setprecision(1) << elapsed_ms << " ms"
                                                      << " (layer " << ctx.layer_idx << ", "
                                                      << ctx.device_id.to_string() << ")");

        // Phase C: Register GPU rebalanced experts in PreparedWeightStore
        if (ctx.prepared_store)
        {
            resolveExpertSlabRefs(ctx, /*create_if_missing=*/true);

            auto register_gpu_rebalance = [&](const std::optional<ExpertSlabRef> &slab_ref,
                                              const std::vector<ITensorGemm *> &engines)
            {
                if (!slab_ref.has_value())
                    return;

                std::vector<ExpertArrival> arrivals;
                for (int e : experts_to_load)
                {
                    if (!engines[e])
                        continue;
                    ExpertArrival arrival;
                    arrival.expert_id = e;
                    arrival.engine = engines[e];
                    // Find shared_ptr lifetime
                    for (const auto &owned : ctx.moe_owned_kernels)
                    {
                        if (owned.get() == engines[e])
                        {
                            arrival.engine_lifetime = owned;
                            break;
                        }
                    }
                    arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
                    arrivals.push_back(std::move(arrival));
                }

                if (!arrivals.empty())
                    ctx.prepared_store->registerArrivedExperts(*slab_ref, arrivals);
            };

            register_gpu_rebalance(ctx.gate_slab_ref, ctx.prepared_gate_gemm);
            register_gpu_rebalance(ctx.up_slab_ref, ctx.prepared_up_gemm);
            register_gpu_rebalance(ctx.down_slab_ref, ctx.prepared_down_gemm);

            LOG_DEBUG("[MoEWeightService::GPU-rebalance] Phase C: Registered "
                      << (count * 3) << " rebalanced expert engines in PreparedWeightStore");
        }

        return true;
    }

    // =========================================================================
    // GPU pipeline path (LoadOrchestrator: raw H2D + GPU repack)
    // =========================================================================

    bool MoEExpertWeightService::prepareGemmEnginesGPU(MoEWeightContext &ctx)
    {
        using Clock = std::chrono::high_resolution_clock;
        const auto t_start = Clock::now();

        const int num_experts = ctx.num_experts;
        const int gpu_ordinal = ctx.device_id.is_cuda()
                                    ? ctx.device_id.cuda_ordinal()
                                    : ctx.device_id.rocm_ordinal();

        // Build list of local expert indices (same as old batch path).
        std::vector<int> local_experts;
        {
            const bool use_mask = !ctx.expert_mask.empty();
            const int local_start = ctx.local_expert_start;
            const int local_count = (ctx.local_expert_count < 0)
                                        ? num_experts
                                        : ctx.local_expert_count;
            const int local_end = local_start + local_count;
            if (use_mask)
            {
                for (int e = 0; e < num_experts; ++e)
                    if (ctx.expert_mask[e])
                        local_experts.push_back(e);
            }
            else
            {
                for (int e = local_start; e < local_end; ++e)
                    local_experts.push_back(e);
            }
        }
        const int local_count = static_cast<int>(local_experts.size());
        if (local_count == 0)
        {
            LOG_DEBUG("[MoEWeightService::GPU] No local experts to prepare on "
                      << ctx.device_id.to_string() << " (layer " << ctx.layer_idx << ")");
            return true;
        }

        // Get backend
        IBackend *backend = getBackendFor(ctx.device_id);
        if (!backend)
        {
            LOG_ERROR("[MoEWeightService::GPU] No backend for " << ctx.device_id.to_string());
            return false;
        }

        // Weight groups: gate, up, down — each with local_count expert views.
        struct WeightGroup
        {
            const char *label;
            std::vector<std::shared_ptr<TensorBase>> &views;
            std::vector<ITensorGemm *> &out_gemms;
            std::shared_ptr<void> &out_lifetime;
        };
        WeightGroup groups[] = {
            {"gate", ctx.expert_gate_views, ctx.prepared_gate_gemm, ctx.moe_packed_gate_lifetime},
            {"up", ctx.expert_up_views, ctx.prepared_up_gemm, ctx.moe_packed_up_lifetime},
            {"down", ctx.expert_down_views, ctx.prepared_down_gemm, ctx.moe_packed_down_lifetime},
        };

        // Create one LoadOrchestrator for ALL expert weights (3 groups × local_count).
        // Single VRAM allocation, pipelined H2D + GPU repack.
        auto orchestrator = std::make_shared<LoadOrchestrator>(backend);
        orchestrator->addDevice(gpu_ordinal);

        size_t max_raw_bytes = 0;
        size_t total_planned = 0;

        // Phase 1: Plan all expert weights
        for (auto &grp : groups)
        {
            for (int idx = 0; idx < local_count; ++idx)
            {
                const int e = local_experts[idx];
                const auto &view = grp.views[e];
                if (!view)
                {
                    LOG_ERROR("[MoEWeightService::GPU] Null view for expert " << e
                                                                              << " in " << grp.label);
                    return false;
                }

                auto *unpackable = dynamic_cast<IINT8Unpackable *>(view.get());
                const NativeVnniFormatInfo *vnni = unpackable ? unpackable->vnniFormatInfo() : nullptr;
                if (!vnni)
                {
                    LOG_ERROR("[MoEWeightService::GPU] Expert " << e << " " << grp.label
                                                                << " has no VNNI format info — cannot use GPU repack");
                    return false;
                }

#ifdef HAVE_ROCM
                if (ctx.device_id.is_rocm() && vnni->codebook_id >= 11 && vnni->codebook_id <= 17)
                {
                    if (!rocm::ensureIQGridTablesInitialized(gpu_ordinal))
                    {
                        LOG_ERROR("[MoEWeightService::GPU] Failed to initialize ROCm IQ grid tables for "
                                  << ctx.device_id.to_string());
                        return false;
                    }
                }
#endif

                const int N = static_cast<int>(view->rows());
                const int K = static_cast<int>(view->cols());
                const size_t raw_bytes = quantizedViewRawBytes(*view);
                if (raw_bytes == 0)
                {
                    LOG_ERROR("[MoEWeightService::GPU] Could not determine raw byte size for expert "
                              << e << " " << grp.label);
                    return false;
                }

                // Unique name per expert per group
                const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

                orchestrator->planWeight(gpu_ordinal, slot_name, N, K,
                                         vnni->payload_bytes, vnni->is_asymmetric,
                                         vnni->has_emins, raw_bytes);
                max_raw_bytes = std::max(max_raw_bytes, raw_bytes);
                ++total_planned;
            }
        }

        LOG_DEBUG("[MoEWeightService::GPU] Planned " << total_planned << " expert weights for "
                                                    << ctx.device_id.to_string() << " (layer " << ctx.layer_idx << ")");

        // Phase 2: Allocate VRAM pool + pinned ring buffer
        const auto &rocm_cfg = debugEnv().rocm;
        const int repack_streams = std::clamp(rocm_cfg.repack_streams, 1, 8);
        const size_t staging_budget_bytes = rocm_cfg.repack_budget_mb > 0
                                                ? static_cast<size_t>(rocm_cfg.repack_budget_mb) * 1024ULL * 1024ULL
                                                : 0;
        const size_t staging_slot_bytes = staging_budget_bytes > 0
                                              ? std::min(max_raw_bytes,
                                                         std::max<size_t>(1, staging_budget_bytes /
                                                                                 static_cast<size_t>(repack_streams)))
                                              : max_raw_bytes;
        orchestrator->allocate(staging_slot_bytes, repack_streams);

        // Phase 3: Create weight jobs
        for (auto &grp : groups)
        {
            for (int idx = 0; idx < local_count; ++idx)
            {
                const int e = local_experts[idx];
                const auto &view = grp.views[e];

                auto *unpackable = dynamic_cast<IINT8Unpackable *>(view.get());
                const NativeVnniFormatInfo *vnni = unpackable->vnniFormatInfo();

                auto repack_fmt = codebookIdToRepackFormat(vnni->codebook_id, vnni->is_superblock);
                if (!repack_fmt)
                {
                    LOG_ERROR("[MoEWeightService::GPU] Unsupported repack format for expert " << e
                                                                                              << " " << grp.label << " (codebook=" << static_cast<int>(vnni->codebook_id)
                                                                                              << ", superblock=" << vnni->is_superblock << ")");
                    return false;
                }

                const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

                WeightJob job;
                job.name = slot_name;
                job.host_raw_data = view->raw_data();
                job.raw_bytes = quantizedViewRawBytes(*view);
                if (!job.host_raw_data)
                {
                    LOG_ERROR("[MoEWeightService::GPU] Expert " << e << " "
                                                                << grp.label << " has null raw host data for " << job.raw_bytes
                                                                << " bytes on " << ctx.device_id.to_string()
                                                                << " (layer " << ctx.layer_idx << ")");
                    return false;
                }
                job.format = *repack_fmt;
                job.N = static_cast<int>(view->rows());
                job.K = static_cast<int>(view->cols());
                job.is_asymmetric = vnni->is_asymmetric;

                orchestrator->addWeightJob(gpu_ordinal, job);
            }
        }

        // Phase 4: Execute pipeline (pipelined H2D + GPU repack)
        orchestrator->load();

        auto *pool = orchestrator->getPool(gpu_ordinal);
        if (!pool)
        {
            LOG_ERROR("[MoEWeightService::GPU] Pool not found after load");
            return false;
        }

        // Phase 5: Create per-expert GEMM kernels from pool slots
        for (auto &grp : groups)
        {
            for (int idx = 0; idx < local_count; ++idx)
            {
                const int e = local_experts[idx];
                const auto &view = grp.views[e];
                const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);

                auto slot = pool->getSlot(slot_name);
                if (!slot)
                {
                    LOG_ERROR("[MoEWeightService::GPU] No slot for " << slot_name);
                    return false;
                }

                auto *unpackable = dynamic_cast<IINT8Unpackable *>(view.get());
                const NativeVnniFormatInfo *vnni = unpackable->vnniFormatInfo();
                const int N = static_cast<int>(view->rows());
                const int K = static_cast<int>(view->cols());
                const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);

                std::shared_ptr<ITensorGemm> kernel;

#ifdef HAVE_CUDA
                if (ctx.device_id.is_cuda())
                {
                    kernel = std::make_shared<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                        N, K, gpu_ordinal,
                        slot->d_native_vnni_payload,
                        static_cast<uint16_t *>(slot->d_native_vnni_scales),
                        static_cast<uint16_t *>(slot->d_native_vnni_mins),
                        static_cast<uint32_t *>(slot->d_native_vnni_emins),
                        vnni->codebook_id, blocks_per_row,
                        orchestrator); // lifetime: keeps VRAM pool alive
                }
#endif
#ifdef HAVE_ROCM
                if (ctx.device_id.is_rocm())
                {
                    kernel = std::make_shared<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                        N, K, gpu_ordinal,
                        slot->d_native_vnni_payload,
                        slot->d_native_vnni_scales,
                        slot->d_native_vnni_mins,
                        slot->d_native_vnni_emins,
                        vnni->codebook_id, blocks_per_row,
                        orchestrator); // lifetime: keeps VRAM pool alive
                }
#endif

                if (!kernel)
                {
                    LOG_ERROR("[MoEWeightService::GPU] Failed to create kernel for " << slot_name);
                    return false;
                }

                grp.out_gemms[e] = kernel.get();
                ctx.moe_owned_kernels.push_back(std::move(kernel));
            }
        }

        // Phase 6: Release orchestrator staging resources (pinned ring).
        // The VRAM pool stays alive via the shared_ptr in each kernel's lifetime_owner_.
        orchestrator->finalize();

        // Phase 7: Mark experts as prepared in the payload provider
        if (ctx.payload_provider)
        {
            for (int e : local_experts)
                ctx.payload_provider->markExpertPrepared(ctx.layer_idx, e);
        }

        const auto elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        LOG_DEBUG("[MoEWeightService::GPU] " << (local_count * 3) << "/" << (num_experts * 3)
                                            << " expert GEMM engines prepared via GPU pipeline in "
                                            << std::fixed << std::setprecision(1) << elapsed_ms << " ms"
                                            << " (layer " << ctx.layer_idx << ", "
                                            << ctx.device_id.to_string() << ")");

        // Release mmap pages for raw expert weights (now repacked on GPU).
        if (ctx.advise_raw_pages_after_prepare)
        {
            size_t released = 0;
            if (ctx.gate_exps && ctx.gate_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(ctx.gate_exps->raw_data(), ctx.gate_exps->size_bytes());
            if (ctx.up_exps && ctx.up_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(ctx.up_exps->raw_data(), ctx.up_exps->size_bytes());
            if (ctx.down_exps && ctx.down_exps->is_mmap_data())
                released += MmapRegion::adviseDontneedRange(ctx.down_exps->raw_data(), ctx.down_exps->size_bytes());
            if (released > 0)
                LOG_DEBUG("[MoEWeightService] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after GPU repack");
        }

        // ── Phase C: Register GPU expert engines in PreparedWeightStore ──────
        if (ctx.prepared_store)
        {
            auto register_gpu_slab = [&](WeightRole role,
                                         const std::vector<ITensorGemm *> &engines) -> ExpertSlabRef
            {
                ExpertSlabDescriptor desc = makeExpertSlabDescriptor(ctx, role);

                auto slab_ref = ctx.prepared_store->registerExpertSlab(desc);

                std::vector<ExpertArrival> arrivals;
                arrivals.reserve(local_experts.size());
                for (int e : local_experts)
                {
                    if (!engines[e])
                        continue;
                    ExpertArrival arrival;
                    arrival.expert_id = e;
                    arrival.engine = engines[e];
                    // Find the shared_ptr lifetime in moe_owned_kernels
                    for (const auto &owned : ctx.moe_owned_kernels)
                    {
                        if (owned.get() == engines[e])
                        {
                            arrival.engine_lifetime = owned;
                            break;
                        }
                    }
                    arrival.derivation = WeightDerivationKind::ExpertSlice;
                    arrivals.push_back(std::move(arrival));
                }

                ctx.prepared_store->registerArrivedExperts(slab_ref, arrivals);
                return slab_ref;
            };

            ctx.gate_slab_ref = register_gpu_slab(WeightRole::MoEExpertGate, ctx.prepared_gate_gemm);
            ctx.up_slab_ref = register_gpu_slab(WeightRole::MoEExpertUp, ctx.prepared_up_gemm);
            ctx.down_slab_ref = register_gpu_slab(WeightRole::MoEExpertDown, ctx.prepared_down_gemm);

            LOG_DEBUG("[MoEWeightService::GPU] Phase C: Registered " << (local_count * 3)
                                                                     << " GPU expert engines in PreparedWeightStore ("
                                                                     << ctx.prepared_store->expertSlabCount() << " slabs, "
                                                                     << ctx.prepared_store->totalPopulatedExperts() << " populated)");
        }

        return true;
    }

    // =========================================================================
    // GPU-direct expert transfer (GPU↔GPU, same packed format)
    // =========================================================================

    size_t GpuDirectStagedExpertArrivals::activationCount() const
    {
        size_t count = 0;
        for (const auto &projection : projections)
        {
            if (projection.activation.has_value())
                ++count;
        }
        return count;
    }

    std::vector<int> GpuDirectStagedExpertArrivals::expertIds() const
    {
        std::vector<int> ids;
        ids.reserve(projections.size());
        for (const auto &projection : projections)
        {
            if (projection.expert_id >= 0)
                ids.push_back(projection.expert_id);
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        return ids;
    }

    bool GpuDirectTransferSlotProjection::valid() const
    {
        return expert_id >= 0 &&
               (role == WeightRole::MoEExpertGate ||
                role == WeightRole::MoEExpertUp ||
                role == WeightRole::MoEExpertDown) &&
               staged.valid() &&
               N > 0 &&
               K > 0 &&
               blocks_per_row > 0 &&
               payload_bytes_per_block > 0 &&
               staged.n == N &&
               staged.k == K &&
               staged.blocks_per_row == blocks_per_row &&
               staged.payload_bytes_per_block == payload_bytes_per_block &&
               staged.is_asymmetric == is_asymmetric &&
               staged.has_emins == has_emins &&
               staged.codebook_id == codebook_id &&
               transfer_slot_lifetime != nullptr;
    }

    size_t GpuDirectTransferSlotArrivals::totalBytes() const
    {
        size_t bytes = 0;
        for (const auto &projection : projections)
            bytes += projection.bytes();
        return bytes;
    }

    std::vector<int> GpuDirectTransferSlotArrivals::expertIds() const
    {
        std::vector<int> ids;
        ids.reserve(projections.size());
        for (const auto &projection : projections)
        {
            if (projection.expert_id >= 0)
                ids.push_back(projection.expert_id);
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        return ids;
    }

    std::vector<std::shared_ptr<void>> GpuDirectTransferSlotArrivals::transferSlotLifetimes() const
    {
        std::vector<std::shared_ptr<void>> lifetimes;
        lifetimes.reserve(projections.size() + completion.transient_lifetimes.size());
        for (const auto &projection : projections)
            appendUniqueLifetime(lifetimes, projection.transfer_slot_lifetime);
        for (const auto &lifetime : completion.transient_lifetimes)
            appendUniqueLifetime(lifetimes, lifetime);
        return lifetimes;
    }

    std::vector<GpuExpertStagedActivation> MoEExpertWeightService::activationBatchForStagedArrivals(
        const GpuDirectStagedExpertArrivals &arrivals)
    {
        std::vector<GpuExpertStagedActivation> batch;
        batch.reserve(arrivals.projections.size());
        for (const auto &projection : arrivals.projections)
        {
            if (projection.activation.has_value())
                batch.push_back(*projection.activation);
        }
        return batch;
    }

    bool MoEExpertWeightService::activateGpuDirectTransferSlotArrivals(
        MoEWeightContext &ctx,
        const GpuDirectTransferSlotArrivals &arrivals,
        void *activation_stream,
        GpuDirectStagedExpertArrivals *activated_arrivals,
        GpuDirectTransferCompletion *completion)
    {
        using Clock = std::chrono::steady_clock;
        if (activated_arrivals)
            *activated_arrivals = GpuDirectStagedExpertArrivals{};
        if (completion)
            *completion = GpuDirectTransferCompletion{};

        if (arrivals.empty())
            return true;

        if (!activation_stream)
        {
            LOG_ERROR("[MoEWeightService] Transfer-slot activation requires an explicit stream for "
                      << ctx.device_id.to_string() << " layer " << ctx.layer_idx);
            return false;
        }
        if (!activated_arrivals)
        {
            LOG_ERROR("[MoEWeightService] Transfer-slot activation requires an output arrival carrier");
            return false;
        }
        if (arrivals.device_id.is_valid() && arrivals.device_id != ctx.device_id)
        {
            LOG_ERROR("[MoEWeightService] Refusing transfer-slot activation for mismatched device: arrivals="
                      << arrivals.device_id.to_string()
                      << " ctx=" << ctx.device_id.to_string());
            return false;
        }
        if (arrivals.layer_idx >= 0 && arrivals.layer_idx != ctx.layer_idx)
        {
            LOG_ERROR("[MoEWeightService] Refusing transfer-slot activation for mismatched layer: arrivals="
                      << arrivals.layer_idx << " ctx=" << ctx.layer_idx);
            return false;
        }
        if (!ctx.device_id.is_gpu())
        {
            LOG_ERROR("[MoEWeightService] Transfer-slot activation requires a GPU destination, got "
                      << ctx.device_id.to_string());
            return false;
        }
        if (!ctx.gpu_direct_slot_pool || !*ctx.gpu_direct_slot_pool)
        {
            LOG_ERROR("[MoEWeightService] Transfer-slot activation requires an existing GPU expert slot pool"
                      << " on " << ctx.device_id.to_string()
                      << " layer " << ctx.layer_idx);
            return false;
        }

        IBackend *backend = getBackendFor(ctx.device_id);
        if (!backend)
        {
            LOG_ERROR("[MoEWeightService] No backend available for transfer-slot activation on "
                      << ctx.device_id.to_string());
            return false;
        }

        const int device_ordinal = gpuOrdinalFor(ctx.device_id);
        if (device_ordinal < 0)
        {
            LOG_ERROR("[MoEWeightService] Invalid GPU ordinal for transfer-slot activation on "
                      << ctx.device_id.to_string());
            return false;
        }

        for (const auto &projection : arrivals.projections)
        {
            if (!projection.valid())
            {
                LOG_ERROR("[MoEWeightService] Invalid transfer-slot projection for layer "
                          << ctx.layer_idx << " expert=" << projection.expert_id);
                return false;
            }
            if (projection.expert_id >= ctx.num_experts)
            {
                LOG_ERROR("[MoEWeightService] Transfer-slot projection expert id out of range: "
                          << projection.expert_id << " num_experts=" << ctx.num_experts);
                return false;
            }
        }

        uint64_t wait_event_ns = 0;
        if (arrivals.completion.valid())
        {
            if (arrivals.completion.device_id != ctx.device_id ||
                arrivals.completion.device_ordinal != device_ordinal)
            {
                LOG_ERROR("[MoEWeightService] Transfer-slot staging completion targets "
                          << arrivals.completion.device_id.to_string()
                          << " ordinal=" << arrivals.completion.device_ordinal
                          << " but activation is on " << ctx.device_id.to_string()
                          << " ordinal=" << device_ordinal
                          << " layer=" << ctx.layer_idx);
                return false;
            }

            const auto wait_start = Clock::now();
            if (!backend->streamWaitEvent(
                    activation_stream,
                    arrivals.completion.ready_event.get(),
                    device_ordinal))
            {
                LOG_ERROR("[MoEWeightService] Failed to make activation stream wait on transfer-slot staging event"
                          << " layer=" << ctx.layer_idx
                          << " device=" << ctx.device_id.to_string());
                return false;
            }
            wait_event_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     Clock::now() - wait_start)
                                                     .count());
        }

        auto *slot_pool = ctx.gpu_direct_slot_pool->get();
        const auto arrival_expert_ids = arrivals.expertIds();
        const size_t available_active_slots = slot_pool->availableSlots();
        if (available_active_slots < arrival_expert_ids.size())
        {
            LOG_ERROR("[MoEWeightService] Transfer-slot activation lacks active-slot capacity for layer "
                      << ctx.layer_idx << " on " << ctx.device_id.to_string()
                      << ": requested=" << arrival_expert_ids.size()
                      << " available=" << available_active_slots
                      << " capacity=" << slot_pool->activeCapacity()
                      << " used=" << slot_pool->usedSlots());
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_transfer_slot_activation_capacity_rejects",
                1.0,
                "rebalance",
                ctx.device_id.to_string(),
                {{"layer", std::to_string(ctx.layer_idx)}});
            return false;
        }

        std::unordered_map<int, GpuExpertSlotPool::AcquiredSlot> active_slots;
        for (int expert_id : arrival_expert_ids)
        {
            auto acquired = slot_pool->acquire(expert_id);
            if (!acquired.has_value())
            {
                LOG_ERROR("[MoEWeightService] Failed to acquire active GPU expert slot for staged expert "
                          << expert_id << " layer " << ctx.layer_idx
                          << " on " << ctx.device_id.to_string());
                return false;
            }
            active_slots.emplace(expert_id, std::move(*acquired));
        }

        GpuDirectStagedExpertArrivals out;
        out.device_id = ctx.device_id;
        out.source_device = arrivals.source_device;
        out.layer_idx = ctx.layer_idx;
        out.projections.reserve(arrivals.projections.size());

        std::vector<GpuExpertStagedActivation> activation_batch;
        activation_batch.reserve(arrivals.projections.size());
        size_t activation_bytes = 0;
        uint64_t wrap_ns = 0;

        for (const auto &projection : arrivals.projections)
        {
            auto active_it = active_slots.find(projection.expert_id);
            if (active_it == active_slots.end())
                return false;

            const char *label = gpuDirectRoleLabel(projection.role);
            if (!label)
                return false;

            auto slot_it = std::find_if(
                active_it->second.projections.begin(),
                active_it->second.projections.end(),
                [&](const GpuExpertSlotPool::ProjectionSlot &slot)
                {
                    return slot.spec.label == label;
                });
            if (slot_it == active_it->second.projections.end())
            {
                LOG_ERROR("[MoEWeightService] Active GPU expert slot for expert "
                          << projection.expert_id << " lacks " << label
                          << " projection on layer " << ctx.layer_idx);
                return false;
            }

            auto active_desc = packedDescFromSlot(
                slot_it->slot,
                projection.N,
                projection.K,
                projection.blocks_per_row,
                projection.codebook_id,
                projection.payload_bytes_per_block,
                projection.is_asymmetric,
                projection.has_emins);
            if (!gpuExpertPackedDescriptorsCompatible(projection.staged, active_desc))
            {
                LOG_ERROR("[MoEWeightService] Transfer-slot and active-slot descriptors are incompatible for "
                          << label << " expert " << projection.expert_id
                          << " layer " << ctx.layer_idx);
                return false;
            }

            GpuExpertStagedActivation activation;
            activation.staged = projection.staged;
            activation.active = active_desc;
            activation_bytes += active_desc.totalBytes();
            activation_batch.push_back(activation);

            const auto wrap_start = Clock::now();
            auto kernel = createGpuDirectKernelFromSlot(
                ctx.device_id,
                device_ordinal,
                slot_it->slot,
                projection.N,
                projection.K,
                projection.blocks_per_row,
                projection.codebook_id,
                active_it->second.lifetime);
            wrap_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 Clock::now() - wrap_start)
                                                 .count());
            if (!kernel)
            {
                LOG_ERROR("[MoEWeightService] Failed to create active-slot GEMM wrapper for "
                          << label << " expert " << projection.expert_id
                          << " layer " << ctx.layer_idx
                          << " on " << ctx.device_id.to_string());
                return false;
            }

            GpuDirectStagedExpertProjection staged_projection;
            staged_projection.expert_id = projection.expert_id;
            staged_projection.role = projection.role;
            staged_projection.engine = kernel.get();
            staged_projection.engine_lifetime = std::move(kernel);
            staged_projection.activation = activation;
            out.projections.push_back(std::move(staged_projection));
        }

        const auto activation_start = Clock::now();
        if (!GPUExpertTransfer::activateStagedExperts(
                activation_batch,
                ctx.device_id,
                activation_stream))
        {
            LOG_ERROR("[MoEWeightService] Transfer-slot activation batch failed for "
                      << activation_batch.size()
                      << " projection(s) layer " << ctx.layer_idx
                      << " dst=" << ctx.device_id.to_string());
            return false;
        }
        const uint64_t activation_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      Clock::now() - activation_start)
                                      .count());

        int completion_events = 0;
        if (completion)
        {
            void *ready_event = backend->createEvent(device_ordinal);
            if (!ready_event ||
                !backend->recordEvent(ready_event, device_ordinal, activation_stream))
            {
                if (ready_event)
                    backend->destroyEvent(ready_event, device_ordinal);
                LOG_ERROR("[MoEWeightService] Failed to record transfer-slot activation completion event"
                          << " layer " << ctx.layer_idx
                          << " on " << ctx.device_id.to_string());
                return false;
            }

            completion->device_id = ctx.device_id;
            completion->device_ordinal = device_ordinal;
            completion->ready_event = eventOwnerFor(backend, device_ordinal, ready_event);
            completion->source_ready_event = arrivals.completion.ready_event;
            completion->transfer_stream = std::shared_ptr<void>(activation_stream, [](void *) {});
            completion->transient_lifetimes = arrivals.transferSlotLifetimes();
            out.completion = *completion;
            completion_events = 1;
        }

        PerfStatsCollector::Tags tags{
            {"layer", std::to_string(ctx.layer_idx)},
            {"dst", ctx.device_id.to_string()}};
        if (arrivals.source_device.has_value())
            tags.emplace("src", arrivals.source_device->to_string());
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_activation_count",
                                       static_cast<double>(activation_batch.size()),
                                       "rebalance", ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_activation_batches",
                                       activation_batch.empty() ? 0.0 : 1.0,
                                       "rebalance", ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_activation_bytes",
                                       static_cast<double>(activation_bytes),
                                       "rebalance", ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_activation_completion_events",
                                       static_cast<double>(completion_events),
                                       "rebalance", ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_slot_activation_wait",
                                           wait_event_ns,
                                           "rebalance", ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_slot_activation_copy",
                                           activation_ns,
                                           "rebalance", ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_slot_activation_wrap",
                                           wrap_ns,
                                           "rebalance", ctx.device_id.to_string(), tags);

        *activated_arrivals = std::move(out);
        return true;
    }

    bool MoEExpertWeightService::installActivatedGpuDirectArrivals(
        MoEWeightContext &ctx,
        const GpuDirectStagedExpertArrivals &arrivals)
    {
        if (arrivals.empty())
            return true;
        if (arrivals.device_id.is_valid() && arrivals.device_id != ctx.device_id)
        {
            LOG_ERROR("[MoEWeightService] Refusing staged GPU arrival install for mismatched device: arrivals="
                      << arrivals.device_id.to_string()
                      << " ctx=" << ctx.device_id.to_string());
            return false;
        }
        if (arrivals.layer_idx >= 0 && arrivals.layer_idx != ctx.layer_idx)
        {
            LOG_ERROR("[MoEWeightService] Refusing staged GPU arrival install for mismatched layer: arrivals="
                      << arrivals.layer_idx << " ctx=" << ctx.layer_idx);
            return false;
        }

        auto vector_for_role = [&](WeightRole role) -> std::vector<ITensorGemm *> *
        {
            if (role == WeightRole::MoEExpertGate)
                return &ctx.prepared_gate_gemm;
            if (role == WeightRole::MoEExpertUp)
                return &ctx.prepared_up_gemm;
            if (role == WeightRole::MoEExpertDown)
                return &ctx.prepared_down_gemm;
            return nullptr;
        };

        auto slab_ref_for_role = [&](WeightRole role) -> std::optional<ExpertSlabRef> *
        {
            if (role == WeightRole::MoEExpertGate)
                return &ctx.gate_slab_ref;
            if (role == WeightRole::MoEExpertUp)
                return &ctx.up_slab_ref;
            if (role == WeightRole::MoEExpertDown)
                return &ctx.down_slab_ref;
            return nullptr;
        };

        auto registry_role_for = [](WeightRole role) -> std::optional<ExpertGemmRegistry::WeightRole>
        {
            if (role == WeightRole::MoEExpertGate)
                return ExpertGemmRegistry::WeightRole::GATE;
            if (role == WeightRole::MoEExpertUp)
                return ExpertGemmRegistry::WeightRole::UP;
            if (role == WeightRole::MoEExpertDown)
                return ExpertGemmRegistry::WeightRole::DOWN;
            return std::nullopt;
        };

        auto remember_owner = [&](const std::shared_ptr<ITensorGemm> &owner)
        {
            if (!owner)
                return;
            const ITensorGemm *raw = owner.get();
            auto it = std::find_if(
                ctx.moe_owned_kernels.begin(),
                ctx.moe_owned_kernels.end(),
                [&](const std::shared_ptr<ITensorGemm> &existing)
                {
                    return existing && existing.get() == raw;
                });
            if (it == ctx.moe_owned_kernels.end())
                ctx.moe_owned_kernels.push_back(owner);
        };

        std::vector<ExpertArrival> gate_arrivals;
        std::vector<ExpertArrival> up_arrivals;
        std::vector<ExpertArrival> down_arrivals;
        auto arrivals_for_role = [&](WeightRole role) -> std::vector<ExpertArrival> *
        {
            if (role == WeightRole::MoEExpertGate)
                return &gate_arrivals;
            if (role == WeightRole::MoEExpertUp)
                return &up_arrivals;
            if (role == WeightRole::MoEExpertDown)
                return &down_arrivals;
            return nullptr;
        };

        std::vector<int> installed_experts;
        installed_experts.reserve(arrivals.projections.size());

        if (ctx.prepared_store)
            resolveExpertSlabRefs(ctx, /*create_if_missing=*/true);

        for (const auto &projection : arrivals.projections)
        {
            if (!projection.valid())
            {
                LOG_ERROR("[MoEWeightService] Invalid staged GPU-direct projection for layer "
                          << ctx.layer_idx << " expert=" << projection.expert_id);
                return false;
            }
            if (projection.expert_id >= ctx.num_experts)
            {
                LOG_ERROR("[MoEWeightService] Staged GPU-direct projection expert id out of range: "
                          << projection.expert_id << " num_experts=" << ctx.num_experts);
                return false;
            }

            auto *engines = vector_for_role(projection.role);
            if (!engines || projection.expert_id >= static_cast<int>(engines->size()))
            {
                LOG_ERROR("[MoEWeightService] Staged GPU-direct projection role/vector mismatch for expert "
                          << projection.expert_id << " layer " << ctx.layer_idx);
                return false;
            }

            (*engines)[projection.expert_id] = projection.engine;
            remember_owner(projection.engine_lifetime);
            installed_experts.push_back(projection.expert_id);

            if (ctx.expert_registry)
            {
                auto registry_role = registry_role_for(projection.role);
                if (registry_role.has_value())
                {
                    ctx.expert_registry->replaceEngine(
                        ctx.device_id,
                        ctx.layer_idx,
                        projection.expert_id,
                        *registry_role,
                        projection.engine,
                        projection.engine_lifetime);
                }
            }

            if (auto *role_arrivals = arrivals_for_role(projection.role))
            {
                ExpertArrival arrival;
                arrival.expert_id = projection.expert_id;
                arrival.engine = projection.engine;
                arrival.engine_lifetime = projection.engine_lifetime;
                arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
                arrival.source_device = arrivals.source_device;
                if (arrivals.completion.valid())
                {
                    auto store_completion = arrivals.completion;
                    store_completion.transient_lifetimes.clear();
                    arrival.gpu_direct_completion = std::move(store_completion);
                }
                role_arrivals->push_back(std::move(arrival));
            }
        }

        if (ctx.payload_provider)
        {
            std::sort(installed_experts.begin(), installed_experts.end());
            installed_experts.erase(
                std::unique(installed_experts.begin(), installed_experts.end()),
                installed_experts.end());
            for (int expert_id : installed_experts)
                ctx.payload_provider->markExpertPrepared(ctx.layer_idx, expert_id);
        }

        if (ctx.prepared_store)
        {
            auto register_role_arrivals = [&](WeightRole role, const std::vector<ExpertArrival> &role_arrivals)
            {
                if (role_arrivals.empty())
                    return;
                auto *slab_ref = slab_ref_for_role(role);
                if (slab_ref && slab_ref->has_value())
                    ctx.prepared_store->registerArrivedExperts(**slab_ref, role_arrivals);
            };

            register_role_arrivals(WeightRole::MoEExpertGate, gate_arrivals);
            register_role_arrivals(WeightRole::MoEExpertUp, up_arrivals);
            register_role_arrivals(WeightRole::MoEExpertDown, down_arrivals);
        }

        return true;
    }

    bool MoEExpertWeightService::stageExpertsGPUDirectToTransferSlots(
        const MoEWeightContext &src_ctx,
        MoEWeightContext &dst_ctx,
        const std::vector<int> &expert_ids,
        int layer_idx,
        void *source_producer_stream,
        GpuDirectTransferSlotArrivals *staged_arrivals,
        std::vector<int> *satisfied_expert_ids,
        size_t active_arrival_capacity,
        size_t staging_pool_capacity,
        std::vector<std::shared_ptr<GpuExpertTransferStagingPool>> *transfer_staging_pools)
    {
        using Clock = std::chrono::steady_clock;
        const auto t_start = Clock::now();
        if (staged_arrivals)
            *staged_arrivals = GpuDirectTransferSlotArrivals{};

        if (satisfied_expert_ids)
            satisfied_expert_ids->clear();

        if (expert_ids.empty())
            return true;

        if (!staged_arrivals)
        {
            LOG_ERROR("[MoEWeightService] GPU-direct transfer-slot staging requires an output carrier");
            return false;
        }

        if (!src_ctx.device_id.is_gpu() || !dst_ctx.device_id.is_gpu() ||
            src_ctx.device_id.type != dst_ctx.device_id.type)
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct transfer-slot staging requires same-backend GPU contexts "
                      << "(src=" << src_ctx.device_id.to_string()
                      << " dst=" << dst_ctx.device_id.to_string() << ")");
            return false;
        }

        if (!source_producer_stream)
        {
            LOG_ERROR("[MoEWeightService] GPU-direct transfer-slot staging requires an explicit source producer stream for "
                      << src_ctx.device_id.to_string());
            return false;
        }

        IBackend *backend = getBackendFor(dst_ctx.device_id);
        IBackend *source_backend = getBackendFor(src_ctx.device_id);
        if (!backend || !source_backend)
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct transfer-slot staging missing backend"
                      << " src=" << src_ctx.device_id.to_string()
                      << " dst=" << dst_ctx.device_id.to_string());
            return false;
        }

        const int dst_gpu_ordinal = gpuOrdinalFor(dst_ctx.device_id);
        const int src_gpu_ordinal = gpuOrdinalFor(src_ctx.device_id);
        if (dst_gpu_ordinal < 0 || src_gpu_ordinal < 0)
            return false;

        struct WeightGroup
        {
            const char *label;
            WeightRole role;
            const std::vector<ITensorGemm *> &src_gemms;
            const std::vector<ITensorGemm *> &dst_gemms;
            std::vector<std::shared_ptr<TensorBase>> &dst_views;
        };

        WeightGroup groups[] = {
            {"gate", WeightRole::MoEExpertGate, src_ctx.prepared_gate_gemm, dst_ctx.prepared_gate_gemm, dst_ctx.expert_gate_views},
            {"up", WeightRole::MoEExpertUp, src_ctx.prepared_up_gemm, dst_ctx.prepared_up_gemm, dst_ctx.expert_up_views},
            {"down", WeightRole::MoEExpertDown, src_ctx.prepared_down_gemm, dst_ctx.prepared_down_gemm, dst_ctx.expert_down_views},
        };
        constexpr size_t group_count = sizeof(groups) / sizeof(groups[0]);

        auto source_slab_ref_for = [&](WeightRole role) -> const std::optional<ExpertSlabRef> &
        {
            if (role == WeightRole::MoEExpertGate)
                return src_ctx.gate_slab_ref;
            if (role == WeightRole::MoEExpertUp)
                return src_ctx.up_slab_ref;
            return src_ctx.down_slab_ref;
        };

        auto stored_engine_for = [&](const MoEWeightContext &ctx,
                                     WeightRole role,
                                     const std::optional<ExpertSlabRef> &cached_ref,
                                     int expert_id) -> ITensorGemm *
        {
            if (!ctx.prepared_store || expert_id < 0 || expert_id >= ctx.num_experts)
                return nullptr;
            if (cached_ref.has_value())
                return ctx.prepared_store->expertGemmKernel(*cached_ref, expert_id);

            auto found = ctx.prepared_store->findExpertSlab(makeExpertSlabDescriptor(ctx, role));
            if (!found.has_value())
                return nullptr;
            return ctx.prepared_store->expertGemmKernel(*found, expert_id);
        };

        auto source_engine_for = [&](const WeightGroup &grp, int expert_id) -> ITensorGemm *
        {
            if (expert_id >= 0 &&
                expert_id < static_cast<int>(grp.src_gemms.size()) &&
                grp.src_gemms[expert_id])
            {
                return grp.src_gemms[expert_id];
            }
            return stored_engine_for(src_ctx, grp.role, source_slab_ref_for(grp.role), expert_id);
        };

        auto destination_already_live = [&](int expert_id) -> bool
        {
            if (expert_id < 0 ||
                expert_id >= static_cast<int>(dst_ctx.prepared_gate_gemm.size()) ||
                expert_id >= static_cast<int>(dst_ctx.prepared_up_gemm.size()) ||
                expert_id >= static_cast<int>(dst_ctx.prepared_down_gemm.size()))
            {
                return false;
            }
            return dst_ctx.prepared_gate_gemm[expert_id] &&
                   dst_ctx.prepared_up_gemm[expert_id] &&
                   dst_ctx.prepared_down_gemm[expert_id];
        };

        std::vector<int> already_satisfied;
        already_satisfied.reserve(expert_ids.size());
        std::vector<int> staged_satisfied;
        staged_satisfied.reserve(expert_ids.size());

        auto publish_ids = [&](const std::vector<int> &ids)
        {
            if (satisfied_expert_ids)
            {
                *satisfied_expert_ids = ids;
                std::sort(satisfied_expert_ids->begin(), satisfied_expert_ids->end());
                satisfied_expert_ids->erase(
                    std::unique(satisfied_expert_ids->begin(), satisfied_expert_ids->end()),
                    satisfied_expert_ids->end());
            }
        };

        auto publish_already_satisfied = [&]()
        {
            publish_ids(already_satisfied);
        };

        auto publish_success = [&]()
        {
            std::vector<int> all = already_satisfied;
            all.insert(all.end(), staged_satisfied.begin(), staged_satisfied.end());
            publish_ids(all);
        };

        std::vector<int> experts_to_stage;
        experts_to_stage.reserve(expert_ids.size());
        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= dst_ctx.num_experts)
            {
                publish_already_satisfied();
                return false;
            }
            if (destination_already_live(expert_id))
            {
                already_satisfied.push_back(expert_id);
                continue;
            }
            experts_to_stage.push_back(expert_id);
        }

        if (experts_to_stage.empty())
        {
            publish_success();
            return true;
        }
        const size_t requested_active_capacity =
            std::max(experts_to_stage.size(), active_arrival_capacity);
        const size_t requested_staging_capacity =
            std::max(experts_to_stage.size(), staging_pool_capacity);

        auto source_descriptor_for = [&](const WeightGroup &grp,
                                         int expert_id,
                                         DeviceNativeVNNIMatrixDesc &out) -> bool
        {
            ITensorGemm *source_engine = source_engine_for(grp, expert_id);
            if (!source_engine)
            {
                LOG_DEBUG("[MoEWeightService] GPU-direct transfer-slot staging: source missing "
                          << grp.label << " engine for expert " << expert_id
                          << " layer " << layer_idx);
                return false;
            }
            if (!source_engine->exportNativeVNNIMatrixDesc(out) || !out.valid())
            {
                LOG_DEBUG("[MoEWeightService] GPU-direct transfer-slot staging: source "
                          << grp.label << " engine for expert " << expert_id
                          << " layer " << layer_idx
                          << " cannot export NativeVNNI descriptor");
                return false;
            }
            return true;
        };

        auto vnni_info_for = [&](const WeightGroup &grp,
                                 int expert_id) -> const NativeVnniFormatInfo *
        {
            if (expert_id < 0 || expert_id >= static_cast<int>(grp.dst_views.size()) ||
                !grp.dst_views[expert_id])
            {
                return nullptr;
            }
            auto *unpackable = dynamic_cast<IINT8Unpackable *>(grp.dst_views[expert_id].get());
            return unpackable ? unpackable->vnniFormatInfo() : nullptr;
        };

        auto validate_optional_destination_view =
            [&](const WeightGroup &grp,
                int expert_id,
                const DeviceNativeVNNIMatrixDesc &src_desc,
                uint8_t payload_bytes_per_block,
                uint8_t is_asymmetric,
                uint8_t has_emins) -> bool
        {
            if (expert_id < 0 || expert_id >= static_cast<int>(grp.dst_views.size()) ||
                !grp.dst_views[static_cast<size_t>(expert_id)])
            {
                return true;
            }

            const auto &view = grp.dst_views[static_cast<size_t>(expert_id)];
            const int N = static_cast<int>(view->rows());
            const int K = static_cast<int>(view->cols());
            const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);
            if (src_desc.n != N ||
                src_desc.k != K ||
                src_desc.blocks_per_row != blocks_per_row)
            {
                LOG_DEBUG("[MoEWeightService] GPU-direct transfer-slot staging: descriptor mismatch for "
                          << grp.label << " expert " << expert_id
                          << " layer " << layer_idx);
                return false;
            }

            if (const NativeVnniFormatInfo *vnni = vnni_info_for(grp, expert_id);
                vnni &&
                (src_desc.codebook_id != vnni->codebook_id ||
                 payload_bytes_per_block != static_cast<uint8_t>(vnni->payload_bytes) ||
                 (is_asymmetric != 0) != vnni->is_asymmetric ||
                 (has_emins != 0) != vnni->has_emins))
            {
                LOG_DEBUG("[MoEWeightService] GPU-direct transfer-slot staging: destination "
                          << grp.label << " view format disagrees with source descriptor for expert "
                          << expert_id << " layer " << layer_idx);
                return false;
            }
            return true;
        };

        std::vector<int> experts_to_copy;
        experts_to_copy.reserve(experts_to_stage.size());
        for (int expert_id : experts_to_stage)
        {
            bool can_copy = true;
            for (const auto &grp : groups)
            {
                DeviceNativeVNNIMatrixDesc src_desc{};
                if (!source_descriptor_for(grp, expert_id, src_desc))
                {
                    can_copy = false;
                    break;
                }

                uint8_t payload_bytes_per_block = 0;
                uint8_t is_asymmetric = 0;
                uint8_t has_emins = 0;
                if (!deviceMoEProjectionFormat(
                        src_desc,
                        payload_bytes_per_block,
                        is_asymmetric,
                        has_emins))
                {
                    LOG_DEBUG("[MoEWeightService] GPU-direct transfer-slot staging: source "
                              << grp.label << " descriptor for expert " << expert_id
                              << " layer " << layer_idx
                              << " has unsupported NativeVNNI codebook");
                    can_copy = false;
                    break;
                }

                if (!validate_optional_destination_view(
                        grp,
                        expert_id,
                        src_desc,
                        payload_bytes_per_block,
                        is_asymmetric,
                        has_emins))
                {
                    can_copy = false;
                    break;
                }
            }
            if (can_copy)
                experts_to_copy.push_back(expert_id);
        }

        if (experts_to_copy.empty())
        {
            publish_success();
            return true;
        }

        auto make_slot_pool_specs = [&](int sample_expert)
            -> std::optional<std::vector<GpuExpertSlotPool::ProjectionSpec>>
        {
            std::vector<GpuExpertSlotPool::ProjectionSpec> specs;
            specs.reserve(group_count);
            for (const auto &grp : groups)
            {
                if (sample_expert < 0 || sample_expert >= dst_ctx.num_experts)
                    return std::nullopt;

                DeviceNativeVNNIMatrixDesc src_desc{};
                if (!source_descriptor_for(grp, sample_expert, src_desc))
                    return std::nullopt;

                uint8_t payload_bytes_per_block = 0;
                uint8_t is_asymmetric = 0;
                uint8_t has_emins = 0;
                if (!deviceMoEProjectionFormat(
                        src_desc,
                        payload_bytes_per_block,
                        is_asymmetric,
                        has_emins))
                {
                    return std::nullopt;
                }

                if (!validate_optional_destination_view(
                        grp,
                        sample_expert,
                        src_desc,
                        payload_bytes_per_block,
                        is_asymmetric,
                        has_emins))
                {
                    return std::nullopt;
                }

                GpuExpertSlotPool::ProjectionSpec spec;
                spec.label = grp.label;
                spec.N = src_desc.n;
                spec.K = src_desc.k;
                spec.payload_bytes_per_block = static_cast<int>(payload_bytes_per_block);
                spec.is_asymmetric = is_asymmetric != 0;
                spec.has_emins = has_emins != 0;
                spec.codebook_id = src_desc.codebook_id;
                specs.push_back(std::move(spec));
            }
            return specs;
        };

        uint64_t allocation_ns = 0;
        auto ensure_slot_pool = [&]() -> std::shared_ptr<GpuExpertSlotPool>
        {
            if (!dst_ctx.gpu_direct_slot_pool)
                return nullptr;
            if (*dst_ctx.gpu_direct_slot_pool)
                return *dst_ctx.gpu_direct_slot_pool;

            auto specs = make_slot_pool_specs(experts_to_copy.front());
            if (!specs.has_value())
                return nullptr;

            const int capacity = GpuExpertSlotPool::recommendedCapacity(
                dst_ctx.num_experts,
                requested_active_capacity);
            const auto pool_start = Clock::now();
            try
            {
                *dst_ctx.gpu_direct_slot_pool = GpuExpertSlotPool::create(
                    backend,
                    dst_ctx.device_id,
                    dst_gpu_ordinal,
                    layer_idx,
                    capacity,
                    std::move(*specs),
                    gpuDirectRebalanceVramSafetyMarginBytes(),
                    /*transfer_capacity=*/0);
            }
            catch (const std::exception &ex)
            {
                LOG_WARN("[MoEWeightService] GPU-direct transfer-slot pool creation failed for layer "
                         << layer_idx << " on " << dst_ctx.device_id.to_string()
                         << ": " << ex.what());
                *dst_ctx.gpu_direct_slot_pool = nullptr;
                return nullptr;
            }
            allocation_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                       Clock::now() - pool_start)
                                                       .count());
            return *dst_ctx.gpu_direct_slot_pool;
        };

        auto ensure_staging_pool = [&]() -> std::shared_ptr<GpuExpertTransferStagingPool>
        {
            if (!transfer_staging_pools)
                return nullptr;

            auto specs = make_slot_pool_specs(experts_to_copy.front());
            if (!specs.has_value())
                return nullptr;

            for (auto &pool : *transfer_staging_pools)
            {
                if (pool && pool->compatibleWith(*specs))
                    return pool;
            }

            const int requested_capacity = GpuExpertTransferStagingPool::recommendedCapacity(
                dst_ctx.num_experts,
                requested_staging_capacity);
            if (requested_capacity <= 0)
                return nullptr;

            std::vector<int> capacity_attempts;
            for (int capacity = requested_capacity; capacity > 0;)
            {
                if (capacity_attempts.empty() || capacity_attempts.back() != capacity)
                    capacity_attempts.push_back(capacity);
                if (capacity == 1)
                    break;
                capacity = std::max(1, capacity / 2);
            }

            std::string last_error;
            for (int capacity : capacity_attempts)
            {
                const auto pool_start = Clock::now();
                try
                {
                    auto pool = GpuExpertTransferStagingPool::create(
                        backend,
                        dst_ctx.device_id,
                        dst_gpu_ordinal,
                        capacity,
                        *specs,
                        gpuDirectRebalanceVramSafetyMarginBytes());
                    transfer_staging_pools->push_back(pool);
                    allocation_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                               Clock::now() - pool_start)
                                                               .count());
                    if (capacity < requested_capacity)
                    {
                        LOG_WARN("[MoEWeightService] GPU-direct transfer staging pool downshifted on "
                                 << dst_ctx.device_id.to_string()
                                 << " from requested rolling capacity=" << requested_capacity
                                 << " to capacity=" << capacity);
                        PerfStatsCollector::addCounter(
                            "moe_rebalance",
                            "gpu_direct_transfer_staging_pool_capacity_downshifts",
                            1.0,
                            "rebalance",
                            dst_ctx.device_id.to_string(),
                            {{"layer", std::to_string(layer_idx)},
                             {"requested_capacity", std::to_string(requested_capacity)},
                             {"capacity", std::to_string(capacity)}});
                    }
                    return pool;
                }
                catch (const std::exception &ex)
                {
                    last_error = ex.what();
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "gpu_direct_transfer_staging_pool_capacity_downshift_attempts",
                        1.0,
                        "rebalance",
                        dst_ctx.device_id.to_string(),
                        {{"layer", std::to_string(layer_idx)},
                         {"requested_capacity", std::to_string(requested_capacity)},
                         {"capacity", std::to_string(capacity)}});
                    continue;
                }
            }

            LOG_ERROR("[MoEWeightService] GPU-direct transfer staging pool creation failed on "
                      << dst_ctx.device_id.to_string()
                      << " for requested rolling wave capacity=" << requested_capacity
                      << (last_error.empty() ? "" : ": ")
                      << last_error);
            return nullptr;
        };

        struct ScopedGpuDirectTransferStream
        {
            IBackend *dst_backend = nullptr;
            IBackend *src_backend = nullptr;
            DeviceId dst_device;
            int dst_device_ordinal = -1;
            int src_device_ordinal = -1;
            void *source_producer_stream = nullptr;
            void *stream = nullptr;
            void *source_ready_event = nullptr;
            void *completion_event = nullptr;
            std::shared_ptr<void> stream_owner;
            bool enqueue_attempted = false;
            bool synchronized = false;
            bool completion_recorded = false;

            ScopedGpuDirectTransferStream(IBackend *dst_backend_in,
                                          IBackend *src_backend_in,
                                          DeviceId dst_device_in,
                                          int dst_device_ordinal_in,
                                          int src_device_ordinal_in,
                                          void *source_producer_stream_in)
                : dst_backend(dst_backend_in),
                  src_backend(src_backend_in),
                  dst_device(dst_device_in),
                  dst_device_ordinal(dst_device_ordinal_in),
                  src_device_ordinal(src_device_ordinal_in),
                  source_producer_stream(source_producer_stream_in) {}

            bool begin(uint64_t &create_ns, uint64_t &event_ns, int &created_count, int &reuse_count)
            {
                if (!dst_backend || !src_backend || !source_producer_stream)
                    return false;
                stream_owner = reusableGpuDirectTransferStreamFor(
                    dst_device,
                    create_ns,
                    created_count,
                    reuse_count);
                if (!stream_owner)
                    return false;
                stream = stream_owner.get();

                const auto event_start = std::chrono::steady_clock::now();
                source_ready_event = src_backend->createEvent(src_device_ordinal);
                if (!source_ready_event ||
                    !src_backend->recordEvent(source_ready_event,
                                              src_device_ordinal,
                                              source_producer_stream) ||
                    !dst_backend->streamWaitEvent(stream,
                                                  source_ready_event,
                                                  dst_device_ordinal))
                {
                    event_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                          std::chrono::steady_clock::now() - event_start)
                                                          .count());
                    return false;
                }
                event_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                      std::chrono::steady_clock::now() - event_start)
                                                      .count());
                return true;
            }

            std::shared_ptr<void> makeEventOwner(void *event, IBackend *owner_backend, int ordinal)
            {
                return std::shared_ptr<void>(
                    event,
                    [owner_backend, ordinal](void *ptr)
                    {
                        if (ptr && owner_backend)
                            owner_backend->destroyEvent(ptr, ordinal);
                    });
            }

            void markEnqueueAttempted()
            {
                enqueue_attempted = true;
            }

            bool recordCompletion(GpuDirectTransferCompletion &out, uint64_t &event_ns)
            {
                if (!stream || !dst_backend)
                    return false;
                const auto event_start = std::chrono::steady_clock::now();
                completion_event = dst_backend->createEvent(dst_device_ordinal);
                if (!completion_event ||
                    !dst_backend->recordEvent(completion_event, dst_device_ordinal, stream))
                {
                    event_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                          std::chrono::steady_clock::now() - event_start)
                                                          .count());
                    return false;
                }

                out.device_id = dst_device;
                out.device_ordinal = dst_device_ordinal;
                out.ready_event = makeEventOwner(completion_event, dst_backend, dst_device_ordinal);
                out.source_ready_event = makeEventOwner(source_ready_event, src_backend, src_device_ordinal);
                out.transfer_stream = stream_owner;
                completion_event = nullptr;
                source_ready_event = nullptr;
                completion_recorded = true;
                event_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                      std::chrono::steady_clock::now() - event_start)
                                                      .count());
                return true;
            }

            ~ScopedGpuDirectTransferStream()
            {
                if (stream && enqueue_attempted && !synchronized && !completion_recorded && dst_backend)
                    (void)dst_backend->synchronizeStream(stream, dst_device_ordinal);
                if (completion_event && dst_backend)
                    dst_backend->destroyEvent(completion_event, dst_device_ordinal);
                if (source_ready_event && src_backend)
                    src_backend->destroyEvent(source_ready_event, src_device_ordinal);
            }
        };

        uint64_t transfer_stream_create_ns = 0;
        uint64_t transfer_stream_event_ns = 0;
        uint64_t transfer_enqueue_ns = 0;
        size_t staged_bytes = 0;
        int transfer_stream_creation_count = 0;
        int transfer_stream_reuse_count = 0;
        int staged_projection_count = 0;
        int staged_expert_count = 0;

        std::shared_ptr<GpuExpertSlotPool> slot_pool = ensure_slot_pool();
        if (!slot_pool)
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct staging requires a destination active slot pool");
            publish_already_satisfied();
            return false;
        }
        std::shared_ptr<GpuExpertTransferStagingPool> staging_pool = ensure_staging_pool();
        if (!staging_pool)
        {
            LOG_ERROR("[MoEWeightService] GPU-direct staging requires a destination transfer staging pool for layer "
                      << layer_idx << " on " << dst_ctx.device_id.to_string());
            publish_already_satisfied();
            return false;
        }
        const size_t available_staging_slots = staging_pool->availableSlots();
        if (available_staging_slots == 0)
        {
            LOG_ERROR("[MoEWeightService] GPU-direct staging pool lacks capacity for rolling wave on layer "
                      << layer_idx << " on " << dst_ctx.device_id.to_string()
                      << ": requested=" << experts_to_copy.size()
                      << " available=" << available_staging_slots
                      << " capacity=" << staging_pool->capacity()
                      << " used=" << staging_pool->usedSlots());
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_transfer_staging_pool_capacity_rejects",
                1.0,
                "rebalance",
                dst_ctx.device_id.to_string(),
                {{"layer", std::to_string(layer_idx)},
                 {"src", src_ctx.device_id.to_string()},
                 {"dst", dst_ctx.device_id.to_string()}});
            publish_already_satisfied();
            return false;
        }
        if (available_staging_slots < experts_to_copy.size())
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct staging trims rolling wave on layer "
                      << layer_idx << " on " << dst_ctx.device_id.to_string()
                      << ": requested=" << experts_to_copy.size()
                      << " available=" << available_staging_slots
                      << " capacity=" << staging_pool->capacity()
                      << " used=" << staging_pool->usedSlots());
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_transfer_staging_pool_wave_trims",
                1.0,
                "rebalance",
                dst_ctx.device_id.to_string(),
                {{"layer", std::to_string(layer_idx)},
                 {"src", src_ctx.device_id.to_string()},
                 {"dst", dst_ctx.device_id.to_string()}});
            experts_to_copy.resize(available_staging_slots);
        }

        ScopedGpuDirectTransferStream transfer_batch(
            backend,
            source_backend,
            dst_ctx.device_id,
            dst_gpu_ordinal,
            src_gpu_ordinal,
            source_producer_stream);
        if (!transfer_batch.begin(transfer_stream_create_ns,
                                  transfer_stream_event_ns,
                                  transfer_stream_creation_count,
                                  transfer_stream_reuse_count))
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct transfer-slot staging: failed to create async transfer stream for "
                      << dst_ctx.device_id.to_string());
            publish_already_satisfied();
            return false;
        }

        GpuDirectTransferSlotArrivals out;
        out.device_id = dst_ctx.device_id;
        out.source_device = src_ctx.device_id;
        out.layer_idx = layer_idx;
        out.projections.reserve(experts_to_copy.size() * group_count);

        std::vector<std::shared_ptr<void>> transfer_lifetimes;
        transfer_lifetimes.reserve(experts_to_copy.size());

        for (int expert_id : experts_to_copy)
        {
            auto staging_lease = staging_pool->acquire(expert_id);
            if (!staging_lease.has_value())
            {
                LOG_DEBUG("[MoEWeightService] GPU-direct staging could not acquire staging slot for expert "
                          << expert_id << " layer " << layer_idx
                          << " on " << dst_ctx.device_id.to_string());
                publish_already_satisfied();
                return false;
            }
            ++staged_expert_count;
            appendUniqueLifetime(transfer_lifetimes, staging_lease->lifetime);

            for (size_t group_index = 0; group_index < group_count; ++group_index)
            {
                const auto &grp = groups[group_index];
                if (group_index >= staging_lease->projections.size())
                {
                    publish_already_satisfied();
                    return false;
                }

                DeviceNativeVNNIMatrixDesc src_matrix{};
                if (!source_descriptor_for(grp, expert_id, src_matrix))
                {
                    publish_already_satisfied();
                    return false;
                }

                uint8_t payload_bytes_per_block = 0;
                uint8_t is_asymmetric = 0;
                uint8_t has_emins = 0;
                if (!deviceMoEProjectionFormat(
                        src_matrix,
                        payload_bytes_per_block,
                        is_asymmetric,
                        has_emins))
                {
                    publish_already_satisfied();
                    return false;
                }

                auto src_desc = makeGpuExpertPackedDescriptor(
                    src_matrix,
                    payload_bytes_per_block,
                    is_asymmetric != 0,
                    has_emins != 0);
                auto dst_desc = packedDescFromSlot(
                    staging_lease->projections[group_index].slot,
                    src_matrix.n,
                    src_matrix.k,
                    src_matrix.blocks_per_row,
                    src_matrix.codebook_id,
                    payload_bytes_per_block,
                    is_asymmetric != 0,
                    has_emins != 0);

                const auto copy_start = Clock::now();
                transfer_batch.markEnqueueAttempted();
                if (!GPUExpertTransfer::transferExpert(
                        src_desc,
                        dst_desc,
                        src_ctx.device_id,
                        dst_ctx.device_id,
                        transfer_batch.stream))
                {
                    LOG_ERROR("[MoEWeightService] GPU-direct transfer-slot staging failed for "
                              << grp.label << " expert " << expert_id
                              << " layer " << layer_idx
                              << " src=" << src_ctx.device_id.to_string()
                              << " dst=" << dst_ctx.device_id.to_string());
                    publish_already_satisfied();
                    return false;
                }
                transfer_enqueue_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                 Clock::now() - copy_start)
                                                                 .count());

                GpuDirectTransferSlotProjection projection;
                projection.expert_id = expert_id;
                projection.role = grp.role;
                projection.staged = dst_desc;
                projection.N = src_matrix.n;
                projection.K = src_matrix.k;
                projection.blocks_per_row = src_matrix.blocks_per_row;
                projection.payload_bytes_per_block = payload_bytes_per_block;
                projection.is_asymmetric = is_asymmetric != 0;
                projection.has_emins = has_emins != 0;
                projection.codebook_id = src_matrix.codebook_id;
                projection.transfer_slot_lifetime = staging_lease->lifetime;
                out.projections.push_back(std::move(projection));
                staged_bytes += src_desc.totalBytes();
                ++staged_projection_count;
            }

            staged_satisfied.push_back(expert_id);
        }

        if (!transfer_batch.recordCompletion(out.completion, transfer_stream_event_ns))
        {
            LOG_ERROR("[MoEWeightService] GPU-direct transfer-slot staging completion event record failed for layer "
                      << layer_idx << " on " << dst_ctx.device_id.to_string());
            publish_already_satisfied();
            return false;
        }
        out.completion.transient_lifetimes = transfer_lifetimes;

        const uint64_t elapsed_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      Clock::now() - t_start)
                                      .count());

        PerfStatsCollector::Tags tags{
            {"layer", std::to_string(layer_idx)},
            {"src", src_ctx.device_id.to_string()},
            {"dst", dst_ctx.device_id.to_string()}};
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_stage_count",
                                       static_cast<double>(staged_projection_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_stage_experts",
                                       static_cast<double>(staged_expert_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_stage_bytes",
                                       static_cast<double>(staged_bytes),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_stage_stream_creations",
                                       static_cast<double>(transfer_stream_creation_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_stage_stream_reuses",
                                       static_cast<double>(transfer_stream_reuse_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_slot_stage_allocate",
                                           allocation_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_slot_stage_event",
                                           transfer_stream_event_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_slot_stage_enqueue",
                                           transfer_enqueue_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_slot_stage",
                                           elapsed_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);

        *staged_arrivals = std::move(out);
        publish_success();
        return true;
    }

    bool MoEExpertWeightService::transferExpertsGPUDirect(
        const MoEWeightContext &src_ctx,
        MoEWeightContext &dst_ctx,
        const std::vector<int> &expert_ids,
        int layer_idx,
        void *source_producer_stream,
        std::vector<int> *satisfied_expert_ids,
        GpuDirectTransferCompletion *completion)
    {
        using Clock = std::chrono::steady_clock;
        const auto t_start = Clock::now();
        if (completion)
            *completion = GpuDirectTransferCompletion{};

        if (expert_ids.empty())
        {
            if (satisfied_expert_ids)
                satisfied_expert_ids->clear();
            return true;
        }

        if (!src_ctx.device_id.is_gpu() || !dst_ctx.device_id.is_gpu() ||
            src_ctx.device_id.type != dst_ctx.device_id.type)
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct transfer requires same-backend GPU contexts "
                      << "(src=" << src_ctx.device_id.to_string()
                      << " dst=" << dst_ctx.device_id.to_string() << ")");
            return false;
        }

        IBackend *backend = getBackendFor(dst_ctx.device_id);
        if (!backend)
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct transfer: no backend for "
                      << dst_ctx.device_id.to_string());
            return false;
        }

        IBackend *source_backend = getBackendFor(src_ctx.device_id);
        if (!source_backend)
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct transfer: no source backend for "
                      << src_ctx.device_id.to_string());
            return false;
        }

        if (!source_producer_stream)
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct transfer requires an explicit source producer stream for "
                      << src_ctx.device_id.to_string());
            return false;
        }

        const int dst_gpu_ordinal = dst_ctx.device_id.is_cuda()
                                        ? dst_ctx.device_id.cuda_ordinal()
                                        : dst_ctx.device_id.rocm_ordinal();
        const int src_gpu_ordinal = src_ctx.device_id.is_cuda()
                                        ? src_ctx.device_id.cuda_ordinal()
                                        : src_ctx.device_id.rocm_ordinal();
        struct WeightGroup
        {
            const char *label;
            WeightRole role;
            const std::vector<ITensorGemm *> &src_gemms;
            std::vector<std::shared_ptr<TensorBase>> &dst_views;
            std::vector<ITensorGemm *> &dst_gemms;
            std::optional<ExpertSlabRef> &dst_slab_ref;
        };

        WeightGroup groups[] = {
            {"gate", WeightRole::MoEExpertGate, src_ctx.prepared_gate_gemm, dst_ctx.expert_gate_views, dst_ctx.prepared_gate_gemm, dst_ctx.gate_slab_ref},
            {"up", WeightRole::MoEExpertUp, src_ctx.prepared_up_gemm, dst_ctx.expert_up_views, dst_ctx.prepared_up_gemm, dst_ctx.up_slab_ref},
            {"down", WeightRole::MoEExpertDown, src_ctx.prepared_down_gemm, dst_ctx.expert_down_views, dst_ctx.prepared_down_gemm, dst_ctx.down_slab_ref},
        };
        constexpr size_t group_count = sizeof(groups) / sizeof(groups[0]);
        resolveExpertSlabRefs(dst_ctx, /*create_if_missing=*/true);

        auto source_slab_ref_for = [&](WeightRole role) -> const std::optional<ExpertSlabRef> &
        {
            if (role == WeightRole::MoEExpertGate)
                return src_ctx.gate_slab_ref;
            if (role == WeightRole::MoEExpertUp)
                return src_ctx.up_slab_ref;
            return src_ctx.down_slab_ref;
        };

        auto stored_engine_for = [&](const MoEWeightContext &ctx,
                                     WeightRole role,
                                     const std::optional<ExpertSlabRef> &cached_ref,
                                     int expert_id) -> ITensorGemm *
        {
            if (!ctx.prepared_store || expert_id < 0 || expert_id >= ctx.num_experts)
                return nullptr;
            if (cached_ref.has_value())
                return ctx.prepared_store->expertGemmKernel(*cached_ref, expert_id);

            auto found = ctx.prepared_store->findExpertSlab(makeExpertSlabDescriptor(ctx, role));
            if (!found.has_value())
                return nullptr;
            return ctx.prepared_store->expertGemmKernel(*found, expert_id);
        };

        auto stored_engine_lifetime_for = [&](const MoEWeightContext &ctx,
                                              WeightRole role,
                                              const std::optional<ExpertSlabRef> &cached_ref,
                                              int expert_id) -> std::shared_ptr<ITensorGemm>
        {
            if (!ctx.prepared_store || expert_id < 0 || expert_id >= ctx.num_experts)
                return nullptr;
            if (cached_ref.has_value())
                return ctx.prepared_store->expertGemmKernelLifetime(*cached_ref, expert_id);

            auto found = ctx.prepared_store->findExpertSlab(makeExpertSlabDescriptor(ctx, role));
            if (!found.has_value())
                return nullptr;
            return ctx.prepared_store->expertGemmKernelLifetime(*found, expert_id);
        };

        auto source_engine_for = [&](const WeightGroup &grp, int expert_id) -> ITensorGemm *
        {
            if (expert_id >= 0 &&
                expert_id < static_cast<int>(grp.src_gemms.size()) &&
                grp.src_gemms[expert_id])
            {
                return grp.src_gemms[expert_id];
            }
            return stored_engine_for(src_ctx, grp.role, source_slab_ref_for(grp.role), expert_id);
        };

        auto destination_owns_engine = [&](ITensorGemm *raw) -> bool
        {
            if (!raw)
                return false;
            for (const auto &owned : dst_ctx.moe_owned_kernels)
            {
                if (owned && owned.get() == raw)
                    return true;
            }
            return false;
        };

        auto adopt_stored_destination_engine = [&](WeightRole role,
                                                   const std::optional<ExpertSlabRef> &cached_ref,
                                                   int expert_id) -> ITensorGemm *
        {
            auto owner = stored_engine_lifetime_for(dst_ctx, role, cached_ref, expert_id);
            if (owner)
            {
                ITensorGemm *raw = owner.get();
                if (!destination_owns_engine(raw))
                    dst_ctx.moe_owned_kernels.push_back(std::move(owner));
                return raw;
            }
            return stored_engine_for(dst_ctx, role, cached_ref, expert_id);
        };

        auto ensure_destination_owner = [&](WeightRole role,
                                            const std::optional<ExpertSlabRef> &cached_ref,
                                            int expert_id,
                                            ITensorGemm *raw)
        {
            if (!raw || destination_owns_engine(raw))
                return;
            auto owner = stored_engine_lifetime_for(dst_ctx, role, cached_ref, expert_id);
            if (owner && owner.get() == raw)
                dst_ctx.moe_owned_kernels.push_back(std::move(owner));
        };

        auto destinationHasExpert = [&](int expert_id) -> bool
        {
            if (expert_id < 0 ||
                expert_id >= static_cast<int>(dst_ctx.prepared_gate_gemm.size()) ||
                expert_id >= static_cast<int>(dst_ctx.prepared_up_gemm.size()) ||
                expert_id >= static_cast<int>(dst_ctx.prepared_down_gemm.size()))
            {
                return false;
            }

            if (!dst_ctx.prepared_gate_gemm[expert_id])
                dst_ctx.prepared_gate_gemm[expert_id] =
                    adopt_stored_destination_engine(WeightRole::MoEExpertGate, dst_ctx.gate_slab_ref, expert_id);
            if (!dst_ctx.prepared_up_gemm[expert_id])
                dst_ctx.prepared_up_gemm[expert_id] =
                    adopt_stored_destination_engine(WeightRole::MoEExpertUp, dst_ctx.up_slab_ref, expert_id);
            if (!dst_ctx.prepared_down_gemm[expert_id])
                dst_ctx.prepared_down_gemm[expert_id] =
                    adopt_stored_destination_engine(WeightRole::MoEExpertDown, dst_ctx.down_slab_ref, expert_id);

            ensure_destination_owner(WeightRole::MoEExpertGate, dst_ctx.gate_slab_ref,
                                     expert_id, dst_ctx.prepared_gate_gemm[expert_id]);
            ensure_destination_owner(WeightRole::MoEExpertUp, dst_ctx.up_slab_ref,
                                     expert_id, dst_ctx.prepared_up_gemm[expert_id]);
            ensure_destination_owner(WeightRole::MoEExpertDown, dst_ctx.down_slab_ref,
                                     expert_id, dst_ctx.prepared_down_gemm[expert_id]);

            return dst_ctx.prepared_gate_gemm[expert_id] &&
                   dst_ctx.prepared_up_gemm[expert_id] &&
                   dst_ctx.prepared_down_gemm[expert_id];
        };

        std::vector<int> satisfied;
        satisfied.reserve(expert_ids.size());
        auto publish_satisfied = [&]()
        {
            std::sort(satisfied.begin(), satisfied.end());
            satisfied.erase(std::unique(satisfied.begin(), satisfied.end()), satisfied.end());
            if (satisfied_expert_ids)
                *satisfied_expert_ids = satisfied;
        };

        std::vector<int> experts_to_load;
        experts_to_load.reserve(expert_ids.size());
        for (int e : expert_ids)
        {
            if (e < 0 || e >= dst_ctx.num_experts)
            {
                publish_satisfied();
                return false;
            }
            if (destinationHasExpert(e))
            {
                satisfied.push_back(e);
                continue;
            }
            experts_to_load.push_back(e);
        }

        if (experts_to_load.empty())
        {
            publish_satisfied();
            return true;
        }

        auto sourceDescriptorFor = [&](const WeightGroup &grp,
                                       int expert_id,
                                       DeviceNativeVNNIMatrixDesc &out) -> bool
        {
            ITensorGemm *source_engine = source_engine_for(grp, expert_id);
            if (!source_engine)
            {
                LOG_DEBUG("[MoEWeightService] GPU-direct transfer: source missing "
                          << grp.label << " engine for expert " << expert_id
                          << " layer " << layer_idx);
                return false;
            }
            if (!source_engine->exportNativeVNNIMatrixDesc(out) || !out.valid())
            {
                LOG_DEBUG("[MoEWeightService] GPU-direct transfer: source "
                          << grp.label << " engine for expert " << expert_id
                          << " layer " << layer_idx
                          << " cannot export NativeVNNI descriptor");
                return false;
            }
            return true;
        };

        auto vnniInfoFor = [&](const WeightGroup &grp,
                               int expert_id) -> const NativeVnniFormatInfo *
        {
            if (expert_id < 0 || expert_id >= static_cast<int>(grp.dst_views.size()) ||
                !grp.dst_views[expert_id])
            {
                return nullptr;
            }
            auto *unpackable = dynamic_cast<IINT8Unpackable *>(grp.dst_views[expert_id].get());
            return unpackable ? unpackable->vnniFormatInfo() : nullptr;
        };

        std::vector<int> experts_to_copy;
        experts_to_copy.reserve(experts_to_load.size());
        for (int e : experts_to_load)
        {
            bool can_copy = true;
            for (const auto &grp : groups)
            {
                DeviceNativeVNNIMatrixDesc src_desc{};
                if (!sourceDescriptorFor(grp, e, src_desc))
                {
                    can_copy = false;
                    break;
                }

                const NativeVnniFormatInfo *vnni = vnniInfoFor(grp, e);
                if (!vnni)
                {
                    LOG_DEBUG("[MoEWeightService] GPU-direct transfer: destination "
                              << grp.label << " view for expert " << e
                              << " layer " << layer_idx
                              << " has no NativeVNNI format info");
                    can_copy = false;
                    break;
                }

                const int N = static_cast<int>(grp.dst_views[e]->rows());
                const int K = static_cast<int>(grp.dst_views[e]->cols());
                const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);

                if (src_desc.n != N ||
                    src_desc.k != K ||
                    src_desc.blocks_per_row != blocks_per_row ||
                    src_desc.codebook_id != vnni->codebook_id)
                {
                    LOG_DEBUG("[MoEWeightService] GPU-direct transfer: descriptor mismatch for "
                              << grp.label << " expert " << e
                              << " layer " << layer_idx
                              << " src=(" << src_desc.n << "x" << src_desc.k
                              << " bpr=" << src_desc.blocks_per_row
                              << " cb=" << static_cast<int>(src_desc.codebook_id)
                              << ") dst=(" << N << "x" << K
                              << " bpr=" << blocks_per_row
                              << " cb=" << static_cast<int>(vnni->codebook_id) << ")");
                    can_copy = false;
                    break;
                }
            }

            if (can_copy)
                experts_to_copy.push_back(e);
        }

        if (experts_to_copy.empty())
        {
            publish_satisfied();
            return true;
        }

        auto make_slot_pool_specs = [&](int sample_expert)
            -> std::optional<std::vector<GpuExpertSlotPool::ProjectionSpec>>
        {
            std::vector<GpuExpertSlotPool::ProjectionSpec> specs;
            specs.reserve(group_count);
            for (const auto &grp : groups)
            {
                const NativeVnniFormatInfo *vnni = vnniInfoFor(grp, sample_expert);
                if (!vnni)
                    return std::nullopt;
                const int N = static_cast<int>(grp.dst_views[sample_expert]->rows());
                const int K = static_cast<int>(grp.dst_views[sample_expert]->cols());
                GpuExpertSlotPool::ProjectionSpec spec;
                spec.label = grp.label;
                spec.N = N;
                spec.K = K;
                spec.payload_bytes_per_block = vnni->payload_bytes;
                spec.is_asymmetric = vnni->is_asymmetric;
                spec.has_emins = vnni->has_emins;
                spec.codebook_id = vnni->codebook_id;
                specs.push_back(std::move(spec));
            }
            return specs;
        };

        auto ensure_slot_pool = [&](uint64_t &allocation_ns_accum)
            -> std::shared_ptr<GpuExpertSlotPool>
        {
            if (!dst_ctx.gpu_direct_slot_pool)
                return nullptr;
            if (*dst_ctx.gpu_direct_slot_pool)
                return *dst_ctx.gpu_direct_slot_pool;

            auto specs = make_slot_pool_specs(experts_to_copy.front());
            if (!specs.has_value())
                return nullptr;

            const int capacity = GpuExpertSlotPool::recommendedCapacity(
                dst_ctx.num_experts,
                experts_to_copy.size());
            const auto pool_start = Clock::now();
            try
            {
                *dst_ctx.gpu_direct_slot_pool = GpuExpertSlotPool::create(
                    backend,
                    dst_ctx.device_id,
                    dst_gpu_ordinal,
                    layer_idx,
                    capacity,
                    std::move(*specs),
                    gpuDirectRebalanceVramSafetyMarginBytes(),
                    GpuExpertSlotPool::recommendedTransferCapacity(
                        dst_ctx.num_experts,
                        experts_to_copy.size()));
            }
            catch (const std::exception &ex)
            {
                LOG_WARN("[MoEWeightService] GPU-direct slot pool creation failed for layer "
                         << layer_idx << " on " << dst_ctx.device_id.to_string()
                         << ": " << ex.what()
                         << " (falling back to per-expert direct allocations)");
                *dst_ctx.gpu_direct_slot_pool = nullptr;
                return nullptr;
            }
            allocation_ns_accum += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                             Clock::now() - pool_start)
                                                             .count());
            return *dst_ctx.gpu_direct_slot_pool;
        };

        auto create_kernel_from_slot = [&](const WeightVRAMPool::WeightSlot &slot,
                                           const NativeVnniFormatInfo &vnni,
                                           int N,
                                           int K,
                                           uint32_t blocks_per_row,
                                           const std::shared_ptr<void> &lifetime_owner)
            -> std::shared_ptr<ITensorGemm>
        {
#ifdef HAVE_CUDA
            if (dst_ctx.device_id.is_cuda())
            {
                return std::make_shared<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                    N, K, dst_gpu_ordinal,
                    slot.d_native_vnni_payload,
                    static_cast<uint16_t *>(slot.d_native_vnni_scales),
                    static_cast<uint16_t *>(slot.d_native_vnni_mins),
                    static_cast<uint32_t *>(slot.d_native_vnni_emins),
                    vnni.codebook_id, blocks_per_row,
                    lifetime_owner);
            }
#endif
#ifdef HAVE_ROCM
            if (dst_ctx.device_id.is_rocm())
            {
                return std::make_shared<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                    N, K, dst_gpu_ordinal,
                    slot.d_native_vnni_payload,
                    slot.d_native_vnni_scales,
                    slot.d_native_vnni_mins,
                    slot.d_native_vnni_emins,
                    vnni.codebook_id, blocks_per_row,
                    lifetime_owner);
            }
#endif
            return nullptr;
        };

        size_t transferred_bytes = 0;
        size_t transfer_slot_activation_bytes = 0;
        int transferred_projections = 0;
        int transfer_slot_activation_projections = 0;
        int transfer_slot_activation_batches = 0;
        int pooled_experts = 0;
        int unpooled_experts = 0;
        int transfer_slot_experts = 0;
        uint64_t allocation_ns = 0;
        uint64_t transfer_enqueue_ns = 0;
        uint64_t transfer_wait_ns = 0;
        uint64_t transfer_slot_activation_ns = 0;
        uint64_t transfer_stream_create_ns = 0;
        uint64_t transfer_stream_event_ns = 0;
        uint64_t wrapper_ns = 0;
        uint64_t finalize_ns = 0;
        int transfer_stream_count = 0;
        int transfer_stream_creation_count = 0;
        int transfer_stream_reuse_count = 0;
        int transfer_stream_sync_count = 0;
        int transfer_stream_event_wait_count = 0;

        GpuDirectStagedExpertArrivals staged_arrivals;
        staged_arrivals.device_id = dst_ctx.device_id;
        staged_arrivals.source_device = src_ctx.device_id;
        staged_arrivals.layer_idx = layer_idx;
        staged_arrivals.projections.reserve(experts_to_copy.size() * group_count);
        std::vector<std::shared_ptr<void>> transient_transfer_lifetimes;
        transient_transfer_lifetimes.reserve(experts_to_copy.size());

        struct ScopedGpuDirectTransferStream
        {
            IBackend *dst_backend = nullptr;
            IBackend *src_backend = nullptr;
            DeviceId dst_device;
            int dst_device_ordinal = -1;
            int src_device_ordinal = -1;
            void *source_producer_stream = nullptr;
            void *stream = nullptr;
            void *source_ready_event = nullptr;
            void *completion_event = nullptr;
            std::shared_ptr<void> stream_owner;
            bool enqueue_attempted = false;
            bool synchronized = false;
            bool completion_recorded = false;

            ScopedGpuDirectTransferStream(IBackend *dst_backend_in,
                                          IBackend *src_backend_in,
                                          DeviceId dst_device_in,
                                          int dst_device_ordinal_in,
                                          int src_device_ordinal_in,
                                          void *source_producer_stream_in)
                : dst_backend(dst_backend_in),
                  src_backend(src_backend_in),
                  dst_device(dst_device_in),
                  dst_device_ordinal(dst_device_ordinal_in),
                  src_device_ordinal(src_device_ordinal_in),
                  source_producer_stream(source_producer_stream_in) {}

            bool begin(uint64_t &create_ns, uint64_t &event_ns, int &created_count, int &reuse_count)
            {
                if (!dst_backend || !src_backend || !source_producer_stream)
                    return false;

                stream_owner = reusableGpuDirectTransferStreamFor(
                    dst_device,
                    create_ns,
                    created_count,
                    reuse_count);
                if (!stream_owner)
                    return false;
                stream = stream_owner.get();

                const auto event_start = Clock::now();
                source_ready_event = src_backend->createEvent(src_device_ordinal);
                if (!source_ready_event ||
                    !src_backend->recordEvent(source_ready_event,
                                              src_device_ordinal,
                                              source_producer_stream) ||
                    !dst_backend->streamWaitEvent(stream,
                                                  source_ready_event,
                                                  dst_device_ordinal))
                {
                    event_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                          Clock::now() - event_start)
                                                          .count());
                    return false;
                }
                event_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                      Clock::now() - event_start)
                                                      .count());
                return true;
            }

            std::shared_ptr<void> makeEventOwner(void *event, IBackend *owner_backend, int ordinal)
            {
                return std::shared_ptr<void>(
                    event,
                    [owner_backend, ordinal](void *ptr)
                    {
                        if (ptr && owner_backend)
                            owner_backend->destroyEvent(ptr, ordinal);
                    });
            }

            void markEnqueueAttempted()
            {
                enqueue_attempted = true;
            }

            bool recordCompletion(GpuDirectTransferCompletion &out, uint64_t &event_ns)
            {
                if (!stream || !dst_backend)
                    return false;

                const auto event_start = Clock::now();
                completion_event = dst_backend->createEvent(dst_device_ordinal);
                if (!completion_event ||
                    !dst_backend->recordEvent(completion_event, dst_device_ordinal, stream))
                {
                    event_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                          Clock::now() - event_start)
                                                          .count());
                    return false;
                }

                out.device_ordinal = dst_device_ordinal;
                out.ready_event = makeEventOwner(completion_event, dst_backend, dst_device_ordinal);
                out.source_ready_event = makeEventOwner(source_ready_event, src_backend, src_device_ordinal);
                out.transfer_stream = stream_owner;

                completion_event = nullptr;
                source_ready_event = nullptr;
                completion_recorded = true;
                event_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                      Clock::now() - event_start)
                                                      .count());
                return true;
            }

            bool finish(uint64_t &wait_ns)
            {
                if (!stream || !dst_backend)
                    return true;
                const auto wait_start = Clock::now();
                const bool ok = dst_backend->synchronizeStream(stream, dst_device_ordinal);
                wait_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     Clock::now() - wait_start)
                                                     .count());
                synchronized = true;
                return ok;
            }

            ~ScopedGpuDirectTransferStream()
            {
                if (stream && enqueue_attempted && !synchronized && !completion_recorded)
                    (void)dst_backend->synchronizeStream(stream, dst_device_ordinal);
                if (completion_event && dst_backend)
                    dst_backend->destroyEvent(completion_event, dst_device_ordinal);
                if (source_ready_event && src_backend)
                    src_backend->destroyEvent(source_ready_event, src_device_ordinal);
            }
        };

        ScopedGpuDirectTransferStream transfer_batch(
            backend,
            source_backend,
            dst_ctx.device_id,
            dst_gpu_ordinal,
            src_gpu_ordinal,
            source_producer_stream);
        if (!transfer_batch.begin(transfer_stream_create_ns,
                                  transfer_stream_event_ns,
                                  transfer_stream_creation_count,
                                  transfer_stream_reuse_count))
        {
            LOG_DEBUG("[MoEWeightService] GPU-direct transfer: failed to create async transfer stream for "
                      << dst_ctx.device_id.to_string());
            publish_satisfied();
            return false;
        }
        transfer_stream_count = 1;
        transfer_stream_event_wait_count = 1;

        for (int e : experts_to_copy)
        {
            std::optional<GpuExpertSlotPool::AcquiredSlot> pooled_slot;
            std::optional<GpuExpertSlotPool::TransferSlot> transfer_slot;
            std::shared_ptr<GpuExpertSlotPool> slot_pool = ensure_slot_pool(allocation_ns);
            if (slot_pool)
            {
                pooled_slot = slot_pool->acquire(e);
                if (pooled_slot.has_value())
                    transfer_slot = slot_pool->acquireTransferSlot(e);
                if (transfer_slot.has_value())
                    ++transfer_slot_experts;
            }

            std::shared_ptr<LoadOrchestrator> expert_orchestrator;
            WeightVRAMPool *fallback_pool = nullptr;
            if (!pooled_slot.has_value())
            {
                const auto alloc_start = Clock::now();
                expert_orchestrator = std::make_shared<LoadOrchestrator>(backend);
                expert_orchestrator->setVramPreflightSafetyMarginBytes(
                    gpuDirectRebalanceVramSafetyMarginBytes());
                expert_orchestrator->addDevice(dst_gpu_ordinal);

                for (const auto &grp : groups)
                {
                    const NativeVnniFormatInfo *vnni = vnniInfoFor(grp, e);
                    if (!vnni)
                    {
                        publish_satisfied();
                        return false;
                    }

                    const int N = static_cast<int>(grp.dst_views[e]->rows());
                    const int K = static_cast<int>(grp.dst_views[e]->cols());
                    const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);
                    expert_orchestrator->planWeight(
                        dst_gpu_ordinal,
                        slot_name,
                        N,
                        K,
                        vnni->payload_bytes,
                        vnni->is_asymmetric,
                        vnni->has_emins,
                        /*raw_gguf_bytes=*/0);
                }

                expert_orchestrator->allocate(/*pinned_slot_size=*/0, /*num_h2d_streams=*/0);
                fallback_pool = expert_orchestrator->getPool(dst_gpu_ordinal);
                if (!fallback_pool)
                {
                    publish_satisfied();
                    return false;
                }
                allocation_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                           Clock::now() - alloc_start)
                                                           .count());
                ++unpooled_experts;
            }
            else
            {
                ++pooled_experts;
            }

            for (size_t group_index = 0; group_index < group_count; ++group_index)
            {
                auto &grp = groups[group_index];
                DeviceNativeVNNIMatrixDesc src_matrix{};
                if (!sourceDescriptorFor(grp, e, src_matrix))
                {
                    publish_satisfied();
                    return false;
                }

                const NativeVnniFormatInfo *vnni = vnniInfoFor(grp, e);
                if (!vnni)
                {
                    publish_satisfied();
                    return false;
                }

                const int N = static_cast<int>(grp.dst_views[e]->rows());
                const int K = static_cast<int>(grp.dst_views[e]->cols());
                const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);
                WeightVRAMPool::WeightSlot active_slot{};
                WeightVRAMPool::WeightSlot copy_dst_slot{};
                std::shared_ptr<void> lifetime_owner;
                bool copy_via_transfer_slot = false;
                if (pooled_slot.has_value())
                {
                    if (group_index >= pooled_slot->projections.size())
                    {
                        publish_satisfied();
                        return false;
                    }
                    const auto &projection = pooled_slot->projections[group_index];
                    active_slot = projection.slot;
                    copy_dst_slot = active_slot;
                    lifetime_owner = pooled_slot->lifetime;
                    if (transfer_slot.has_value())
                    {
                        if (group_index >= transfer_slot->projections.size())
                        {
                            publish_satisfied();
                            return false;
                        }
                        copy_dst_slot = transfer_slot->projections[group_index].slot;
                        copy_via_transfer_slot = true;
                    }
                }
                else
                {
                    const std::string slot_name = std::string(grp.label) + "_e" + std::to_string(e);
                    auto fallback_slot = fallback_pool ? fallback_pool->getSlot(slot_name) : std::nullopt;
                    if (!fallback_slot)
                    {
                        publish_satisfied();
                        return false;
                    }
                    active_slot = *fallback_slot;
                    copy_dst_slot = active_slot;
                    lifetime_owner = expert_orchestrator;
                }

                DeviceNativeVNNIMatrixDesc dst_matrix{};
                dst_matrix.payload = copy_dst_slot.d_native_vnni_payload;
                dst_matrix.scales = copy_dst_slot.d_native_vnni_scales;
                dst_matrix.mins = copy_dst_slot.d_native_vnni_mins;
                dst_matrix.emins = copy_dst_slot.d_native_vnni_emins;
                dst_matrix.n = N;
                dst_matrix.k = K;
                dst_matrix.blocks_per_row = blocks_per_row;
                dst_matrix.codebook_id = vnni->codebook_id;

                auto src_desc = makeGpuExpertPackedDescriptor(
                    src_matrix,
                    vnni->payload_bytes,
                    vnni->is_asymmetric,
                    vnni->has_emins);
                auto dst_desc = makeGpuExpertPackedDescriptor(
                    dst_matrix,
                    vnni->payload_bytes,
                    vnni->is_asymmetric,
                    vnni->has_emins);

                const auto copy_start = Clock::now();
                transfer_batch.markEnqueueAttempted();
                const bool copied = GPUExpertTransfer::transferExpert(
                    src_desc,
                    dst_desc,
                    src_ctx.device_id,
                    dst_ctx.device_id,
                    transfer_batch.stream);
                transfer_enqueue_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                 Clock::now() - copy_start)
                                                                 .count());
                if (!copied)
                {
                    LOG_ERROR("[MoEWeightService] GPU-direct transfer failed for "
                              << grp.label << " expert " << e
                              << " layer " << layer_idx
                              << " src=" << src_ctx.device_id.to_string()
                              << " dst=" << dst_ctx.device_id.to_string());
                    publish_satisfied();
                    return false;
                }

                if (copy_via_transfer_slot)
                {
                    DeviceNativeVNNIMatrixDesc active_matrix{};
                    active_matrix.payload = active_slot.d_native_vnni_payload;
                    active_matrix.scales = active_slot.d_native_vnni_scales;
                    active_matrix.mins = active_slot.d_native_vnni_mins;
                    active_matrix.emins = active_slot.d_native_vnni_emins;
                    active_matrix.n = N;
                    active_matrix.k = K;
                    active_matrix.blocks_per_row = blocks_per_row;
                    active_matrix.codebook_id = vnni->codebook_id;

                    auto active_desc = makeGpuExpertPackedDescriptor(
                        active_matrix,
                        vnni->payload_bytes,
                        vnni->is_asymmetric,
                        vnni->has_emins);

                    GpuExpertStagedActivation activation;
                    activation.staged = dst_desc;
                    activation.active = active_desc;
                    transfer_slot_activation_bytes += active_desc.totalBytes();
                    ++transfer_slot_activation_projections;
                    auto active_activation = std::move(activation);

                    const auto wrapper_start = Clock::now();
                    auto kernel = create_kernel_from_slot(
                        active_slot,
                        *vnni,
                        N,
                        K,
                        blocks_per_row,
                        lifetime_owner);

                    if (!kernel)
                    {
                        publish_satisfied();
                        return false;
                    }

                    GpuDirectStagedExpertProjection projection;
                    projection.expert_id = e;
                    projection.role = grp.role;
                    projection.engine = kernel.get();
                    projection.engine_lifetime = std::move(kernel);
                    projection.activation = std::move(active_activation);
                    staged_arrivals.projections.push_back(std::move(projection));
                    transferred_bytes += src_desc.totalBytes();
                    ++transferred_projections;
                    wrapper_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                            Clock::now() - wrapper_start)
                                                            .count());
                    continue;
                }

                const auto wrapper_start = Clock::now();
                auto kernel = create_kernel_from_slot(
                    active_slot,
                    *vnni,
                    N,
                    K,
                    blocks_per_row,
                    lifetime_owner);

                if (!kernel)
                {
                    publish_satisfied();
                    return false;
                }

                GpuDirectStagedExpertProjection projection;
                projection.expert_id = e;
                projection.role = grp.role;
                projection.engine = kernel.get();
                projection.engine_lifetime = std::move(kernel);
                staged_arrivals.projections.push_back(std::move(projection));
                transferred_bytes += src_desc.totalBytes();
                ++transferred_projections;
                wrapper_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                        Clock::now() - wrapper_start)
                                                        .count());
            }

            if (transfer_slot.has_value() && transfer_slot->lifetime)
                transient_transfer_lifetimes.push_back(transfer_slot->lifetime);

            if (expert_orchestrator)
            {
                const auto finalize_start = Clock::now();
                expert_orchestrator->finalize();
                finalize_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                         Clock::now() - finalize_start)
                                                         .count());
            }
        }

        auto pending_transfer_slot_activations =
            activationBatchForStagedArrivals(staged_arrivals);
        if (!pending_transfer_slot_activations.empty())
        {
            const auto activation_start = Clock::now();
            transfer_batch.markEnqueueAttempted();
            const bool activated = GPUExpertTransfer::activateStagedExperts(
                pending_transfer_slot_activations,
                dst_ctx.device_id,
                transfer_batch.stream);
            transfer_slot_activation_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - activation_start)
                    .count());
            if (!activated)
            {
                LOG_ERROR("[MoEWeightService] GPU-direct transfer-slot activation batch failed for "
                          << pending_transfer_slot_activations.size()
                          << " projection(s) layer " << layer_idx
                          << " dst=" << dst_ctx.device_id.to_string());
                publish_satisfied();
                return false;
            }
            transfer_slot_activation_batches = 1;
        }

        int transfer_completion_event_count = 0;
        if (completion)
        {
            completion->device_id = dst_ctx.device_id;
            if (!transfer_batch.recordCompletion(*completion, transfer_stream_event_ns))
            {
                LOG_ERROR("[MoEWeightService] GPU-direct transfer completion event record failed for layer "
                          << layer_idx << " on " << dst_ctx.device_id.to_string());
                publish_satisfied();
                return false;
            }
            completion->transient_lifetimes = transient_transfer_lifetimes;
            transfer_completion_event_count = 1;
        }
        else if (!transfer_batch.finish(transfer_wait_ns))
        {
            LOG_ERROR("[MoEWeightService] GPU-direct transfer stream synchronization failed for layer "
                      << layer_idx << " on " << dst_ctx.device_id.to_string());
            publish_satisfied();
            return false;
        }
        transfer_stream_sync_count = completion ? 0 : 1;

        const auto registration_start = Clock::now();
        if (completion && completion->valid())
            staged_arrivals.completion = *completion;
        if (!installActivatedGpuDirectArrivals(dst_ctx, staged_arrivals))
        {
            publish_satisfied();
            return false;
        }
        const uint64_t registration_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      Clock::now() - registration_start)
                                      .count());

        const uint64_t elapsed_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      Clock::now() - t_start)
                                      .count());

        for (int e : experts_to_copy)
        {
            if (!destinationHasExpert(e))
            {
                LOG_ERROR("[MoEWeightService] GPU-direct transfer completed but destination still lacks expert "
                          << e << " layer " << layer_idx << " on " << dst_ctx.device_id.to_string());
                publish_satisfied();
                return false;
            }
            satisfied.push_back(e);
        }

        PerfStatsCollector::Tags tags{
            {"layer", std::to_string(layer_idx)},
            {"src", src_ctx.device_id.to_string()},
            {"dst", dst_ctx.device_id.to_string()}};
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_count",
                                       static_cast<double>(transferred_projections),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_bytes",
                                       static_cast<double>(transferred_bytes),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_slot_pool_experts",
                                       static_cast<double>(pooled_experts),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_unpooled_experts",
                                       static_cast<double>(unpooled_experts),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_experts",
                                       static_cast<double>(transfer_slot_experts),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_activation_count",
                                       static_cast<double>(transfer_slot_activation_projections),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_activation_batches",
                                       static_cast<double>(transfer_slot_activation_batches),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_slot_activation_bytes",
                                       static_cast<double>(transfer_slot_activation_bytes),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_streams",
                                       static_cast<double>(transfer_stream_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_stream_creations",
                                       static_cast<double>(transfer_stream_creation_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_stream_reuses",
                                       static_cast<double>(transfer_stream_reuse_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_stream_syncs",
                                       static_cast<double>(transfer_stream_sync_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_event_waits",
                                       static_cast<double>(transfer_stream_event_wait_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::addCounter("moe_rebalance", "gpu_direct_transfer_completion_events",
                                       static_cast<double>(transfer_completion_event_count),
                                       "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_arrival_allocate",
                                           allocation_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_stream_create",
                                           transfer_stream_create_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_stream_event_wait",
                                           transfer_stream_event_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_enqueue",
                                           transfer_enqueue_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_wait",
                                           transfer_wait_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_slot_activation_copy",
                                           transfer_slot_activation_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer_copy",
                                           transfer_enqueue_ns + transfer_wait_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_arrival_wrap",
                                           wrapper_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_arrival_finalize",
                                           finalize_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_arrival_register",
                                           registration_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);
        PerfStatsCollector::recordTimingNs("moe_rebalance", "gpu_direct_transfer",
                                           elapsed_ns,
                                           "rebalance", dst_ctx.device_id.to_string(), tags);

        LOG_DEBUG("[MoEWeightService] GPU-direct transferred "
                  << experts_to_copy.size() << " experts ("
                  << transferred_projections << " projections, "
                  << transferred_bytes << " bytes, pooled="
                  << pooled_experts << ", unpooled=" << unpooled_experts << ") "
                  << src_ctx.device_id.to_string()
                  << " → " << dst_ctx.device_id.to_string()
                  << " layer " << layer_idx);
        publish_satisfied();
        return true;
    }

} // namespace llaminar2
