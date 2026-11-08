/**
 * @file CPUProjectionWorkspaceContract.cpp
 * @brief Uses the production serial tile policy to price CPU projection buffers.
 *
 * Only source-format adaptation differs from prepared execution. Encodings
 * come from the packer and K trees come from the common serial policy, with
 * physical N reserved separately from an optional logical serial width. No
 * root-local CPUID, anonymous reserve or per-layer live balance exists here.
 */
#include "CPUProjectionWorkspaceContract.h"
#include "CPUNativeVNNITileConfig.h"
#include "CPUNativeVNNIEncoding.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include <climits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Validate matrix dimensions before either floating or quantized declaration. */
        void requireGeometry(const CPUProjectionWorkspaceGeometry &geometry)
        {
            if (geometry.rows <= 0 || geometry.n <= 0 || geometry.n > INT_MAX - 63 ||
                geometry.k <= 0 || geometry.k > INT_MAX - 31 || geometry.serial_n < 0)
                throw std::invalid_argument("CPU projection workspace requires complete representable matrix geometry");
        }
    }

    WorkspaceRequirements CPUProjectionWorkspaceContract::sourceNative(
        std::string_view format, const CPUProjectionWorkspaceGeometry &geometry)
    {
        requireGeometry(geometry);
        if (format == "F32" || format == "F16" || format == "BF16") return {};
        const auto *native = native_vnni_formats::forQuantType(format);
        if (!native) throw std::invalid_argument("CPU projection workspace has an unsupported source format");
        return prepared(cpu::native_vnni::preparedFootprintForFormat(*native), native->codebook_id, geometry);
    }

    WorkspaceRequirements CPUProjectionWorkspaceContract::prepared(
        const cpu::native_vnni::NativeVNNIPreparedFootprint &footprint, std::uint8_t codebook,
        const CPUProjectionWorkspaceGeometry &geometry)
    {
        requireGeometry(geometry);
        if (geometry.workers <= 0 || !geometry.execution.isValid())
            throw std::invalid_argument("CPU projection workspace requires participant-owned worker/cache/ISA geometry");
        if (!footprint.isValid() ||
            (geometry.numerical_policy != CPUProjectionNumericalPolicy::BackendNative &&
             geometry.numerical_policy != CPUProjectionNumericalPolicy::GPUAlignedExpert))
            throw std::invalid_argument("CPU projection workspace requires a known prepared footprint and arithmetic policy");
        if (geometry.execution.projection_abi == CPUProjectionScheduleABI::ARMRegisterK)
        {
            if (!MoEProjectionNumericalContract::ownsOrderedKPartitionTree(codebook) ||
                (geometry.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert && geometry.k % 32 != 0))
                throw std::invalid_argument("Native ARM projection workspace requires a supported source and complete expert K blocks");
            // The same registered implementation owns both ordinary full-K and
            // canonical expert trees in registers. No x86 shared-K slab or
            // anonymous capacity reserve is charged to the native participant.
            return cpuProjectionQ8WorkspaceRequirements(geometry.rows, geometry.k);
        }
        const auto config = cpu::native_vnni::serialTileConfigForPreparedProjection(footprint,
            codebook, geometry.numerical_policy, geometry.serial_n > 0 ? geometry.serial_n : geometry.n,
            geometry.k, geometry.workers, geometry.execution.cache);
        auto requirements = cpuProjectionQ8WorkspaceRequirements(geometry.rows, geometry.k);
        const auto partials = cpuProjectionPartialWorkspaceRequirements(cpuProjectionPartialEnvelopeFloats(
            geometry.rows, geometry.n, config.k_tiles, geometry.workers,
            static_cast<int>(geometry.execution.maximum_native_row_tile)));
        // These are distinct canonical projection names (Q8 versus partials),
        // not grouped-MoE root-layout merges. Keep this pure BOM independent of
        // the unrelated storage-layout materialization/composition machinery.
        requirements.buffers.insert(requirements.buffers.end(), partials.buffers.begin(), partials.buffers.end());
        return requirements;
    }

    WorkspaceRequirements CPUProjectionWorkspaceContract::sourceNative(
        TensorType format, const CPUProjectionWorkspaceGeometry &geometry)
    {
        // Only the two floating-point spellings differ between TensorType and
        // GGUF metadata. Quantized identities use the canonical format catalog.
        const std::string_view name = format == TensorType::FP32 ? "F32" :
            format == TensorType::FP16 ? "F16" : tensorTypeName(format);
        return sourceNative(name, geometry);
    }
}
