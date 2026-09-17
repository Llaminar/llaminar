/**
 * @file KernelFactoryGPUWeights.cpp
 * @brief Shared backend binding for admitted prepared GPU matrix regions.
 *
 * Model loading and bounded planning samples use the same matrix constructors.
 * LoadOrchestrator retains all physical storage and its PMA claims. This factory
 * validates backend identity and each logical subregion before creating an
 * execution handle; it neither loads bytes nor publishes weight readiness.
 * Source codebook identity stays distinct from reusable allocation stride.
 */
#include "KernelFactory.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "tensors/TensorClasses.h"
#ifdef HAVE_CUDA
#include "cuda/gemm/CUDAFloatingPointGemmKernel.h"
#include "cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif
#ifdef HAVE_ROCM
#include "rocm/gemm/ROCmFloatingPointGemmKernel.h"
#include "rocm/gemm/ROCmQuantisedGemmKernel.h"
#endif
#include <limits>
#include <stdexcept>

namespace llaminar::v2::kernels
{
    namespace
    {
        /**
         * @brief Compute a byte count without letting wraparound authorize a pool view.
         * @param left Number of rows/blocks in the requested region.
         * @param right Bytes or blocks per element of that extent.
         * @return Checked product; an absent component has zero bytes.
         * @throws std::overflow_error if the product is not addressable.
         */
        size_t product(size_t left, size_t right)
        {
            if (right && left > std::numeric_limits<size_t>::max() / right)
                throw std::overflow_error("Prepared GPU matrix region overflows addressable bytes");
            return left * right;
        }

        /**
         * @brief Resolve one exact pool subregion without touching device bytes.
         * @param base Stable allocation pointer supplied by the pool owner.
         * @param capacity Bytes owned by this physical region.
         * @param offset First byte of the logical matrix in that region.
         * @param length Logical execution bytes; zero means an absent component.
         * @param slot_name Logical slot for a precise setup-failure diagnostic.
         * @param component Payload, scales, minima or embedded-minima region.
         * @return Borrowed address, or null when no backing component exists.
         * @throws std::out_of_range for missing or insufficient physical storage.
         */
        void *region(void *base, size_t capacity, size_t offset, size_t length,
            const std::string &slot_name, const char *component)
        {
            if (offset > capacity || length > capacity - offset || (length && !base))
                throw std::out_of_range("Prepared GPU matrix slot '" + slot_name + "' " + component +
                    " region: offset=" + std::to_string(offset) + " bytes=" + std::to_string(length) +
                    " exceeds/misses backing capacity=" + std::to_string(capacity));
            return base ? static_cast<uint8_t *>(base) + offset : nullptr;
        }
    }

    std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemmFromGPUWeightPool(
        const llaminar2::TensorBase &tensor, llaminar2::DeviceId device,
        std::shared_ptr<llaminar2::LoadOrchestrator> owner, const std::string &slot_name,
        size_t row_offset, llaminar2::GPUPreparedWeightPoolLayout layout)
    {
        using namespace llaminar2;
        if (!device.is_gpu() || !owner || owner->managedDevice(device.ordinal) != device ||
            slot_name.empty() || tensor.shape().size() != 2 ||
            tensor.rows() == 0 || tensor.cols() == 0 ||
            tensor.rows() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            tensor.cols() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            (layout != GPUPreparedWeightPoolLayout::SourceNative &&
             layout != GPUPreparedWeightPoolLayout::MigrationReusable))
            throw std::invalid_argument("GPU weight-pool GEMM requires matching ownership and valid matrix metadata");
        const auto *pool = owner->getPool(device.ordinal);
        if (!pool || !pool->isAllocated() || pool->deviceId() != device.ordinal)
            throw std::invalid_argument("GPU weight-pool GEMM requires an allocated device pool");
        const auto slot = pool->getSlot(slot_name);
        if (!slot) throw std::invalid_argument("GPU weight-pool GEMM names an absent slot: " + slot_name);
        const int n = static_cast<int>(tensor.rows()), k = static_cast<int>(tensor.cols());
        const auto type = tensor.native_type();
        const bool floating = type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16;
        if (floating)
        {
            const size_t row_bytes = product(static_cast<size_t>(k), type == TensorType::FP32 ? 4 : 2);
            auto *payload = region(slot->d_native_vnni_payload, slot->payload_bytes,
                product(row_offset, row_bytes), product(static_cast<size_t>(n), row_bytes), slot_name, "floating payload");
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                using Kernel = cuda::CUDAFloatingPointGemmKernel;
                const auto precision = type == TensorType::FP32 ? Kernel::Precision::FP32 :
                    type == TensorType::FP16 ? Kernel::Precision::FP16 : Kernel::Precision::BF16;
                return std::make_unique<Kernel>(payload, n, k, device.ordinal, precision, std::move(owner));
            }
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                using Kernel = rocm::ROCmFloatingPointGemmKernel;
                const auto precision = type == TensorType::FP32 ? Kernel::Precision::FP32 :
                    type == TensorType::FP16 ? Kernel::Precision::FP16 : Kernel::Precision::BF16;
                return std::make_unique<Kernel>(payload, n, k, device.ordinal, precision, std::move(owner));
            }
#endif
        }
        else
        {
            const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(&tensor);
            const auto *source = unpackable ? unpackable->vnniFormatInfo() : nullptr;
            if (!source || k % 32 != 0)
                throw std::invalid_argument("GPU weight-pool GEMM requires a supported complete NativeVNNI matrix");
            const auto reusable = layout == GPUPreparedWeightPoolLayout::MigrationReusable
                ? reusableDeviceVnniAllocationFormat(*source) : NativeVnniReusableDeviceAllocationFormat{};
            const auto allocation = layout == GPUPreparedWeightPoolLayout::MigrationReusable
                ? reusable : NativeVnniReusableDeviceAllocationFormat{
                    static_cast<uint8_t>(source->payload_bytes), source->is_asymmetric, source->has_emins};
            const auto blocks_per_row = static_cast<size_t>(k / 32);
            const auto offset_blocks = product(row_offset, blocks_per_row);
            const auto length_blocks = product(static_cast<size_t>(n), blocks_per_row);
            // Both offsets and bounds use the allocation union. Preserve every
            // reserved metadata pointer even when the initial format does not
            // read it: later migration may publish a representation that does.
            auto *payload = static_cast<uint8_t *>(region(slot->d_native_vnni_payload, slot->payload_bytes,
                product(offset_blocks, allocation.payload_bytes_per_block), product(length_blocks, allocation.payload_bytes_per_block),
                slot_name, "quantized payload"));
            auto *scales = static_cast<uint16_t *>(region(slot->d_native_vnni_scales, slot->scales_bytes,
                product(offset_blocks, sizeof(uint16_t)), product(length_blocks, sizeof(uint16_t)), slot_name, "scales"));
            auto *mins = static_cast<uint16_t *>(region(slot->d_native_vnni_mins, slot->mins_bytes,
                allocation.has_mins ? product(offset_blocks, sizeof(uint16_t)) : 0,
                allocation.has_mins ? product(length_blocks, sizeof(uint16_t)) : 0, slot_name, "minima"));
            auto *emins = static_cast<uint32_t *>(region(slot->d_native_vnni_emins, slot->emins_bytes,
                allocation.has_emins ? product(offset_blocks, sizeof(uint32_t)) : 0,
                allocation.has_emins ? product(length_blocks, sizeof(uint32_t)) : 0, slot_name, "embedded minima"));
            const NativeVnniSourceIdentity identity{source->codebook_id, source->is_superblock, true};
            const auto codebook = canonicalDeviceVnniCodebookId(source->codebook_id);
#ifdef HAVE_CUDA
            if (device.is_cuda())
                return std::make_unique<cuda::CUDAQuantisedGemmKernel>(n, k, device.ordinal,
                    payload, scales, mins, emins, codebook, static_cast<uint32_t>(blocks_per_row),
                    std::move(owner), identity, reusable);
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm())
                return std::make_unique<rocm::ROCmQuantisedGemmKernel>(n, k, device.ordinal,
                    payload, scales, mins, emins, codebook, static_cast<uint32_t>(blocks_per_row),
                    std::move(owner), identity, reusable);
#endif
        }
        throw std::invalid_argument("GPU weight-pool GEMM backend was not built");
    }
}
