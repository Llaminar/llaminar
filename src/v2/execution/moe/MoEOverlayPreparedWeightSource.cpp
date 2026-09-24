/**
 * @file MoEOverlayPreparedWeightSource.cpp
 * @brief Validation of live CPU/GPU prepared-weight migration authorities.
 */

#include "MoEOverlayPreparedWeightSource.h"

#include "ExpertTierWeightDeviceLayout.h"
#include "tensors/TensorKernels.h"

#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Store one resolver error without throwing from maintenance. */
        bool rejectPreparedSource(
            std::string *error,
            std::string message) noexcept
        {
            if (error)
                *error = std::move(message);
            return false;
        }

        /** @brief Physical execution format exported by one GPU engine. */
        struct GpuExecutionFormat
        {
            std::uint8_t payload_bytes = 0;
            bool is_asymmetric = false;
            bool has_emins = false;
        };

        /**
         * @brief Recover physical GPU bytes without confusing source identity.
         *
         * A GPU source may either retain its original prepared format or be a
         * normalized promotion from CPU ExpandedInt8. The latter deliberately
         * uses codebook 19/23 while keeping the original GGUF identity in the
         * engine, so both cases must be recognized explicitly.
         */
        bool resolveGpuExecutionFormat(
            const DeviceNativeVNNIMatrixDesc &descriptor,
            const NativeVnniFormatInfo &source,
            GpuExecutionFormat &format,
            std::string *error) noexcept
        {
            const std::uint8_t canonical =
                canonicalDeviceVnniCodebookId(source.codebook_id);
            if (descriptor.codebook_id == canonical)
            {
                format = {
                    .payload_bytes =
                        static_cast<std::uint8_t>(source.payload_bytes),
                    .is_asymmetric = source.is_asymmetric,
                    .has_emins = source.has_emins,
                };
                return true;
            }

            const auto cpu_encoding =
                cpu::native_vnni::preparedEncodingForCodebook(
                    source.codebook_id);
            if (cpu_encoding !=
                cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8)
            {
                return rejectPreparedSource(
                    error,
                    "GPU execution codebook differs from non-normalizable source identity");
            }

            const std::uint8_t normalized =
                source.is_asymmetric
                    ? kNativeVnniExpandedInt8MinCodebook
                    : static_cast<std::uint8_t>(19);
            if (descriptor.codebook_id != normalized)
            {
                return rejectPreparedSource(
                    error,
                    "GPU execution codebook is neither canonical nor the exact CPU-promotion normalization");
            }
            format = {
                .payload_bytes = 32,
                .is_asymmetric = source.is_asymmetric,
                .has_emins = false,
            };
            return true;
        }

        /** @brief Validate optional GPU arrays and reusable-slot capacities. */
        bool validateGpuPhysicalStorage(
            const DeviceNativeVNNIMatrixDesc &descriptor,
            const GpuExecutionFormat &format,
            std::string *error) noexcept
        {
            if (!descriptor.valid() || descriptor.k % 32 != 0 ||
                descriptor.blocks_per_row !=
                    static_cast<std::uint32_t>(descriptor.k / 32))
            {
                return rejectPreparedSource(
                    error,
                    "GPU prepared source has invalid NativeVNNI geometry");
            }
            if (format.is_asymmetric && !descriptor.mins)
            {
                return rejectPreparedSource(
                    error,
                    "GPU prepared source is missing its required minima array");
            }
            if (format.has_emins && !descriptor.emins)
            {
                return rejectPreparedSource(
                    error,
                    "GPU prepared source is missing its required extended-minima array");
            }

            /* Zero allocation fields identify ordinary immutable model weights. */
            if (descriptor.allocation_payload_bytes_per_block != 0 &&
                descriptor.allocation_payload_bytes_per_block <
                    format.payload_bytes)
            {
                return rejectPreparedSource(
                    error,
                    "GPU prepared source payload exceeds its slot allocation capacity");
            }
            if (descriptor.allocation_payload_bytes_per_block != 0 &&
                format.is_asymmetric && !descriptor.allocation_has_mins)
            {
                return rejectPreparedSource(
                    error,
                    "GPU prepared source minima exceed its slot allocation capacity");
            }
            if (descriptor.allocation_payload_bytes_per_block != 0 &&
                format.has_emins && !descriptor.allocation_has_emins)
            {
                return rejectPreparedSource(
                    error,
                    "GPU prepared source extended minima exceed its slot allocation capacity");
            }
            return true;
        }
    } // namespace

    bool MoEOverlayPreparedWeightSource::valid(
        std::string *error) const noexcept
    {
        if (error)
            error->clear();
        if (!engine || !format.valid())
        {
            return rejectPreparedSource(
                error,
                "Prepared migration source lacks retained engine ownership or valid format provenance");
        }

        if (kind == MoEOverlayPreparedWeightSourceKind::CpuNativeVnni)
        {
            if (!format.isNativeVnni() || !device.is_cpu() || !cpu_packed ||
                gpu_packed.valid() || floating.valid() ||
                cpu_packed->N <= 0 || cpu_packed->K <= 0 ||
                cpu_packed->blocks_per_row != cpu_packed->K / 32 ||
                cpu_packed->native_interleaved.empty() ||
                cpu_packed->codebook_id != format.native_vnni.codebook_id ||
                cpu_packed->is_superblock !=
                    format.native_vnni.is_superblock)
            {
                return rejectPreparedSource(
                    error,
                    "CPU prepared migration source disagrees with its device, geometry, or provenance");
            }
            return true;
        }

        if (kind ==
            MoEOverlayPreparedWeightSourceKind::GpuSeparatedNativeVnni)
        {
            if (!format.isNativeVnni() || !device.is_gpu() || cpu_packed ||
                !gpu_packed.valid() || floating.valid())
            {
                return rejectPreparedSource(
                    error,
                    "GPU prepared migration source disagrees with its device or NativeVNNI layout authority");
            }
            return true;
        }

        const bool expects_cpu =
            kind ==
            MoEOverlayPreparedWeightSourceKind::CpuContiguousFloating;
        const bool expects_gpu =
            kind ==
            MoEOverlayPreparedWeightSourceKind::GpuContiguousFloating;
        if (!format.isFloating() || (!expects_cpu && !expects_gpu) ||
            (expects_cpu && !device.is_cpu()) ||
            (expects_gpu && !device.is_gpu()) || cpu_packed ||
            gpu_packed.valid() || !floating.valid() ||
            format.floatingTensorType() !=
                std::optional<TensorType>(floating.type))
        {
            return rejectPreparedSource(
                error,
                "Floating prepared migration source disagrees with its device, precision, or contiguous layout authority");
        }
        return true;
    }

    bool resolveMoEOverlayPreparedWeightSource(
        std::shared_ptr<ITensorGemm> engine,
        DeviceId device,
        MoEOverlayPreparedWeightSource &output,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        output = {};
        if (!engine || (!device.is_cpu() && !device.is_gpu()))
        {
            return rejectPreparedSource(
                error,
                "Prepared migration source requires an owned engine and CPU/GPU participant");
        }

        NativeVnniSourceIdentity source_identity;
        const bool has_native_identity =
            engine->exportNativeVNNISourceIdentity(source_identity) &&
            source_identity.present;
        ContiguousFloatingPointWeightDescriptor floating;
        const bool has_floating =
            engine->exportContiguousFloatingPointWeights(floating) &&
            floating.valid();
        if (has_native_identity == has_floating)
        {
            return rejectPreparedSource(
                error,
                "Prepared migration engine must export exactly one quantized or floating source authority");
        }

        if (has_floating)
        {
            output = {
                .kind = device.is_cpu()
                            ? MoEOverlayPreparedWeightSourceKind::
                                  CpuContiguousFloating
                            : MoEOverlayPreparedWeightSourceKind::
                                  GpuContiguousFloating,
                .device = device,
                .format = ExpertWeightFormat::floating(floating.type),
                .engine = std::move(engine),
                .floating = floating,
            };
            return output.valid(error);
        }

        const NativeVnniFormatInfo *source =
            native_vnni_formats::forSourceIdentity(
                source_identity.codebook_id,
                source_identity.is_superblock);
        if (!source)
        {
            return rejectPreparedSource(
                error,
                "Prepared migration engine exports an ambiguous or unsupported source identity");
        }

        const auto *cpu_packed =
            engine->exportCPUNativeVNNIPackedWeights();
        DeviceNativeVNNIMatrixDesc gpu_descriptor;
        const bool has_gpu_descriptor =
            engine->exportNativeVNNIMatrixDesc(gpu_descriptor);
        if (device.is_cpu())
        {
            if (!cpu_packed || has_gpu_descriptor)
            {
                return rejectPreparedSource(
                    error,
                    "CPU migration engine must export exactly one CPU packed authority");
            }
            output = {
                .kind =
                    MoEOverlayPreparedWeightSourceKind::CpuNativeVnni,
                .device = device,
                .format = ExpertWeightFormat::nativeVnni(source_identity),
                .engine = std::move(engine),
                .cpu_packed = cpu_packed,
            };
            return output.valid(error);
        }

        if (cpu_packed || !has_gpu_descriptor)
        {
            return rejectPreparedSource(
                error,
                "GPU migration engine must export exactly one device descriptor authority");
        }
        GpuExecutionFormat format;
        if (!resolveGpuExecutionFormat(
                gpu_descriptor, *source, format, error) ||
            !validateGpuPhysicalStorage(gpu_descriptor, format, error))
        {
            return false;
        }

        output = {
            .kind =
                MoEOverlayPreparedWeightSourceKind::GpuSeparatedNativeVnni,
            .device = device,
            .format = ExpertWeightFormat::nativeVnni(source_identity),
            .engine = std::move(engine),
            .gpu_packed = makeGpuExpertPackedDescriptor(
                gpu_descriptor,
                format.payload_bytes,
                format.is_asymmetric,
                format.has_emins),
        };
        return output.valid(error);
    }
} // namespace llaminar2
