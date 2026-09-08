/**
 * @file Perf__ExpertTierWeightStreaming.cpp
 * @brief Release benchmark for production ExpertOverlay cross-tier repacking.
 *
 * One source is compiled into separate CUDA and ROCm executables.  Each
 * executable prepares real quantized tensors through the production
 * NativeVNNI packer, proves both conversion directions byte-for-byte, and then
 * measures the exact explicit-stream kernels with and without pinned-host DMA.
 * The default matrix covers every loader-supported quantized format and the
 * two projection geometries used by Qwen3.5/Qwen3.6 MoE experts.
 *
 * Profilers should use `--single-launch` with one format, shape, and direction.
 * That mode performs all allocation and upload work first, launches exactly
 * one selected conversion kernel, and observes completion only on its owning
 * non-default stream.
 */

#include "execution/moe/ExpertTierWeightStream.h"
#include "tensors/TensorClasses.h"
#include "tensors/VnniPackContext.h"
#include "utils/QuantizedVerifierFormats.h"

#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
#include "kernels/cuda/repack/CUDAExpertTierWeightKernels.h"
#include <cuda_runtime.h>
#elif defined(LLAMINAR_EXPERT_TIER_PERF_ROCM)
#include "kernels/rocm/gemm/ROCmWeightPacker.h"
#include "kernels/rocm/repack/ROCmExpertTierWeightKernels.h"
#include <hip/hip_runtime.h>
#else
#error "Select exactly one ExpertTierWeightStreaming benchmark backend"
#endif

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <numa.h>

#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
extern "C"
{
    /** Publish production IQ decode tables before an IQ tier conversion. */
    bool cudaNativeVNNIInitIQGridTables_tuned();
}
#endif

namespace llaminar2
{
    namespace
    {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
        using RuntimeStatus = cudaError_t;
        using RuntimeStream = cudaStream_t;
        using RuntimeEvent = cudaEvent_t;
        constexpr RuntimeStatus kRuntimeSuccess = cudaSuccess;
        constexpr std::string_view kBackendName = "cuda";
#else
        using RuntimeStatus = hipError_t;
        using RuntimeStream = hipStream_t;
        using RuntimeEvent = hipEvent_t;
        constexpr RuntimeStatus kRuntimeSuccess = hipSuccess;
        constexpr std::string_view kBackendName = "rocm";
#endif

        /** Direction of one measured cross-device-type projection. */
        enum class Direction : std::uint8_t
        {
            GpuToCpu,
            CpuToGpu,
        };

        /** Scope of the measured asynchronous transaction. */
        enum class MeasurementMode : std::uint8_t
        {
            Kernel,
            Streaming,
        };

        /** One production expert projection geometry. */
        struct ProjectionShape
        {
            std::string_view name;
            int N = 0;
            int K = 0;
        };

        constexpr ProjectionShape kGateUpShape{"gate_up", 512, 2048};
        constexpr ProjectionShape kDownShape{"down", 2048, 512};

        /** Parsed benchmark policy; all defaults form the exhaustive sweep. */
        struct Options
        {
            std::string format = "all";
            std::string shape = "all";
            std::string direction = "both";
            std::string mode = "both";
            // Both production Qwen expert projections contain 512 CPU units.
            // The production 4 MiB staging lane therefore streams each one as
            // a single chunk for every supported encoding.
            std::uint32_t units_per_chunk = 512;
            int warmup = 5;
            int iterations = 20;
            int samples = 3;
            int device = 0;
            std::string numa = "auto";
            double minimum_kernel_gbps = 5.0;
            double minimum_streaming_gbps = 1.0;
            bool single_launch = false;
            std::optional<std::string> csv_path;
        };

        /** One emitted timing record with enough geometry to reproduce it. */
        struct Measurement
        {
            std::string_view backend;
            int numa_node = -1;
            MeasurementMode mode = MeasurementMode::Kernel;
            Direction direction = Direction::GpuToCpu;
            std::string_view format;
            std::uint8_t source_codebook = 0;
            std::uint8_t physical_codebook = 0;
            ProjectionShape shape;
            std::uint32_t unit_count = 0;
            std::uint32_t units_per_chunk = 0;
            std::uint32_t chunk_count = 0;
            std::uint64_t semantic_bytes = 0;
            double mean_ms = 0.0;
            double gb_per_s = 0.0;
            int iterations = 0;
            int samples = 0;
            double economy_floor_gbps = 0.0;
            bool economical = false;
        };

        /** Convert a backend status to a stable diagnostic string. */
        [[nodiscard]] const char *runtimeErrorString(RuntimeStatus status)
        {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            return cudaGetErrorString(status);
#else
            return hipGetErrorString(status);
#endif
        }

        /** Throw a contextual error instead of continuing with partial setup. */
        void requireRuntime(RuntimeStatus status, std::string_view operation)
        {
            if (status == kRuntimeSuccess)
                return;
            throw std::runtime_error(
                std::string(operation) + ": " + runtimeErrorString(status));
        }

        /** Select the exact accelerator before any stream or allocation exists. */
        void selectDevice(int device)
        {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            requireRuntime(cudaSetDevice(device), "cudaSetDevice");
#else
            requireRuntime(hipSetDevice(device), "hipSetDevice");
#endif
        }

        /** Return the number of devices visible to this isolated runtime. */
        [[nodiscard]] int deviceCount()
        {
            int count = 0;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            requireRuntime(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
#else
            requireRuntime(hipGetDeviceCount(&count), "hipGetDeviceCount");
#endif
            return count;
        }

        /** Resolve the selected accelerator's canonical PCI bus identifier. */
        [[nodiscard]] std::string devicePciBusId(int device)
        {
            char identifier[32]{};
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            requireRuntime(
                cudaDeviceGetPCIBusId(
                    identifier, sizeof(identifier), device),
                "cudaDeviceGetPCIBusId");
#else
            requireRuntime(
                hipDeviceGetPCIBusId(
                    identifier, sizeof(identifier), device),
                "hipDeviceGetPCIBusId");
#endif
            std::string result(identifier);
            std::transform(
                result.begin(),
                result.end(),
                result.begin(),
                [](unsigned char value)
                {
                    return static_cast<char>(std::tolower(value));
                });
            if (result.empty())
                throw std::runtime_error(
                    "runtime returned an empty PCI bus identifier");
            return result;
        }

        /** Own the one non-default transfer stream used by a fixture. */
        class BenchmarkStream final
        {
        public:
            /** Create an explicitly non-blocking runtime stream. */
            BenchmarkStream()
            {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                requireRuntime(
                    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags");
#else
                requireRuntime(
                    hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
                    "hipStreamCreateWithFlags");
#endif
                if (stream_ == nullptr)
                    throw std::runtime_error("runtime returned a null stream");
            }

            /** Destroy the stream after every owned event and buffer is idle. */
            ~BenchmarkStream()
            {
                if (stream_ == nullptr)
                    return;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                (void)cudaStreamDestroy(stream_);
#else
                (void)hipStreamDestroy(stream_);
#endif
            }

            BenchmarkStream(const BenchmarkStream &) = delete;
            BenchmarkStream &operator=(const BenchmarkStream &) = delete;

            /** @return Native stream for runtime event and copy APIs. */
            [[nodiscard]] RuntimeStream native() const noexcept
            {
                return stream_;
            }

            /** @return Opaque non-null stream consumed by production launchers. */
            [[nodiscard]] void *opaque() const noexcept
            {
                return reinterpret_cast<void *>(stream_);
            }

            /** Wait at a benchmark observation boundary on this exact stream. */
            void synchronize() const
            {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                requireRuntime(
                    cudaStreamSynchronize(stream_),
                    "cudaStreamSynchronize");
#else
                requireRuntime(
                    hipStreamSynchronize(stream_),
                    "hipStreamSynchronize");
#endif
            }

        private:
            RuntimeStream stream_ = nullptr;
        };

        /** Own one timing event recorded only on the fixture transfer stream. */
        class BenchmarkEvent final
        {
        public:
            /** Create a timing-capable event before measured work starts. */
            BenchmarkEvent()
            {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                requireRuntime(cudaEventCreate(&event_), "cudaEventCreate");
#else
                requireRuntime(hipEventCreate(&event_), "hipEventCreate");
#endif
            }

            /** Release the event after its final observation. */
            ~BenchmarkEvent()
            {
                if (event_ == nullptr)
                    return;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                (void)cudaEventDestroy(event_);
#else
                (void)hipEventDestroy(event_);
#endif
            }

            BenchmarkEvent(const BenchmarkEvent &) = delete;
            BenchmarkEvent &operator=(const BenchmarkEvent &) = delete;

            /** Record this event after prior work on the exact transfer stream. */
            void record(const BenchmarkStream &stream)
            {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                requireRuntime(
                    cudaEventRecord(event_, stream.native()),
                    "cudaEventRecord");
#else
                requireRuntime(
                    hipEventRecord(event_, stream.native()),
                    "hipEventRecord");
#endif
            }

            /** Wait only at the outer measurement boundary. */
            void synchronize() const
            {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                requireRuntime(
                    cudaEventSynchronize(event_),
                    "cudaEventSynchronize");
#else
                requireRuntime(
                    hipEventSynchronize(event_),
                    "hipEventSynchronize");
#endif
            }

            /** Return elapsed milliseconds between two completed events. */
            [[nodiscard]] static float elapsed(
                const BenchmarkEvent &begin,
                const BenchmarkEvent &end)
            {
                float milliseconds = 0.0f;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                requireRuntime(
                    cudaEventElapsedTime(
                        &milliseconds, begin.event_, end.event_),
                    "cudaEventElapsedTime");
#else
                requireRuntime(
                    hipEventElapsedTime(
                        &milliseconds, begin.event_, end.event_),
                    "hipEventElapsedTime");
#endif
                return milliseconds;
            }

        private:
            RuntimeEvent event_ = nullptr;
        };

        /** Own one exact-capacity device allocation made before timing. */
        template <typename T>
        class DeviceBuffer final
        {
        public:
            /** Allocate `elements` values; the valid empty case stays null. */
            explicit DeviceBuffer(std::size_t elements)
                : elements_(elements)
            {
                if (elements_ == 0)
                    return;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                requireRuntime(
                    cudaMalloc(
                        reinterpret_cast<void **>(&pointer_), bytes()),
                    "cudaMalloc");
#else
                requireRuntime(
                    hipMalloc(
                        reinterpret_cast<void **>(&pointer_), bytes()),
                    "hipMalloc");
#endif
            }

            /** Release the allocation after its owning stream is idle. */
            ~DeviceBuffer()
            {
                if (pointer_ == nullptr)
                    return;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                (void)cudaFree(pointer_);
#else
                (void)hipFree(pointer_);
#endif
            }

            DeviceBuffer(const DeviceBuffer &) = delete;
            DeviceBuffer &operator=(const DeviceBuffer &) = delete;

            /** @return Writable device address, or null for zero elements. */
            [[nodiscard]] T *data() noexcept { return pointer_; }

            /** @return Read-only device address, or null for zero elements. */
            [[nodiscard]] const T *data() const noexcept { return pointer_; }

            /** @return Exact byte capacity. */
            [[nodiscard]] std::size_t bytes() const noexcept
            {
                return elements_ * sizeof(T);
            }

        private:
            T *pointer_ = nullptr;
            std::size_t elements_ = 0;
        };

        /** Own pinned host bytes so streaming timings exercise real async DMA. */
        class PinnedBytes final
        {
        public:
            /** Allocate exact portable pinned storage before any measurement. */
            explicit PinnedBytes(std::size_t bytes)
                : bytes_(bytes)
            {
                if (bytes_ == 0)
                    throw std::invalid_argument(
                        "pinned benchmark allocation must be non-empty");
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                requireRuntime(
                    cudaHostAlloc(
                        reinterpret_cast<void **>(&pointer_),
                        bytes_,
                        cudaHostAllocPortable),
                    "cudaHostAlloc");
#else
                requireRuntime(
                    hipHostMalloc(
                        reinterpret_cast<void **>(&pointer_),
                        bytes_,
                        hipHostMallocPortable),
                    "hipHostMalloc");
#endif
            }

            /** Release pinned storage after all DMA observations complete. */
            ~PinnedBytes()
            {
                if (pointer_ == nullptr)
                    return;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
                (void)cudaFreeHost(pointer_);
#else
                (void)hipHostFree(pointer_);
#endif
            }

            PinnedBytes(const PinnedBytes &) = delete;
            PinnedBytes &operator=(const PinnedBytes &) = delete;

            /** @return Writable pinned address. */
            [[nodiscard]] std::uint8_t *data() noexcept { return pointer_; }

            /** @return Read-only pinned address. */
            [[nodiscard]] const std::uint8_t *data() const noexcept
            {
                return pointer_;
            }

            /** @return Exact byte capacity. */
            [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

        private:
            std::uint8_t *pointer_ = nullptr;
            std::size_t bytes_ = 0;
        };

        /** Enqueue host-to-device bytes on the selected non-default stream. */
        void copyHostToDeviceAsync(
            void *destination,
            const void *source,
            std::size_t bytes,
            const BenchmarkStream &stream)
        {
            if (bytes == 0)
                return;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            requireRuntime(
                cudaMemcpyAsync(
                    destination,
                    source,
                    bytes,
                    cudaMemcpyHostToDevice,
                    stream.native()),
                "cudaMemcpyAsync(H2D)");
#else
            requireRuntime(
                hipMemcpyAsync(
                    destination,
                    source,
                    bytes,
                    hipMemcpyHostToDevice,
                    stream.native()),
                "hipMemcpyAsync(H2D)");
#endif
        }

        /** Enqueue device-to-host bytes on the selected non-default stream. */
        void copyDeviceToHostAsync(
            void *destination,
            const void *source,
            std::size_t bytes,
            const BenchmarkStream &stream)
        {
            if (bytes == 0)
                return;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            requireRuntime(
                cudaMemcpyAsync(
                    destination,
                    source,
                    bytes,
                    cudaMemcpyDeviceToHost,
                    stream.native()),
                "cudaMemcpyAsync(D2H)");
#else
            requireRuntime(
                hipMemcpyAsync(
                    destination,
                    source,
                    bytes,
                    hipMemcpyDeviceToHost,
                    stream.native()),
                "hipMemcpyAsync(D2H)");
#endif
        }

        /** Fill device storage on the selected stream during fixture setup. */
        void zeroDeviceAsync(
            void *destination,
            std::size_t bytes,
            const BenchmarkStream &stream)
        {
            if (bytes == 0)
                return;
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            requireRuntime(
                cudaMemsetAsync(destination, 0, bytes, stream.native()),
                "cudaMemsetAsync");
#else
            requireRuntime(
                hipMemsetAsync(destination, 0, bytes, stream.native()),
                "hipMemsetAsync");
#endif
        }

        /**
         * Pack a real source tensor into the common CUDA/ROCm separated layout.
         * This is the same `packVnniBlock` authority used by weight preparation.
         */
        [[nodiscard]] HostGpuExpertPackedProjection packGpuProjection(
            const TensorBase &tensor)
        {
            const auto *unpackable =
                dynamic_cast<const IINT8Unpackable *>(&tensor);
            if (unpackable == nullptr || unpackable->vnniFormatInfo() == nullptr)
                throw std::invalid_argument(
                    "benchmark format is not NativeVNNI-packable");

            const NativeVnniFormatInfo &format =
                *unpackable->vnniFormatInfo();
            const int N = static_cast<int>(tensor.rows());
            const int K = static_cast<int>(tensor.cols());
            if (N <= 0 || K <= 0 || (K % 32) != 0)
                throw std::invalid_argument(
                    "benchmark projection is not a complete NativeVNNI geometry");

            HostGpuExpertPackedProjection projection;
            projection.N = N;
            projection.K = K;
            projection.blocks_per_row = static_cast<std::uint32_t>(K / 32);
            projection.source_codebook_id = format.codebook_id;
            projection.codebook_id =
                canonicalDeviceVnniCodebookId(format.codebook_id);
            projection.payload_bytes_per_block =
                static_cast<std::uint8_t>(format.payload_bytes);
            projection.is_asymmetric = format.is_asymmetric;
            projection.is_superblock = format.is_superblock;
            projection.has_emins = format.has_emins;

            const std::size_t blocks =
                static_cast<std::size_t>(N) * projection.blocks_per_row;
            projection.payload.resize(blocks * format.payload_bytes);
            projection.scales.resize(blocks);
            if (format.is_asymmetric)
                projection.mins.resize(blocks);
            if (format.has_emins)
                projection.emins.resize(blocks);

            VnniPackContext context{};
            context.N = N;
            context.K = K;
            context.blocks_per_row = K / 32;
            context.payload_bytes = format.payload_bytes;
            context.payload_array = projection.payload.data();
            context.scales_array = projection.scales.data();
            context.mins_array =
                format.is_asymmetric ? projection.mins.data() : nullptr;
            context.emins_array =
                format.has_emins ? projection.emins.data() : nullptr;

            for (int n = 0; n < N; ++n)
            {
                for (int kb = 0; kb < context.blocks_per_row; ++kb)
                    unpackable->packVnniBlock(context, n, n, kb);
            }

            std::string error;
            if (!projection.valid(&error))
                throw std::runtime_error(
                    "invalid production GPU projection: " + error);
            return projection;
        }

        /** Upload an entire typed vector before measured work is enqueued. */
        template <typename T>
        void uploadVector(
            DeviceBuffer<T> &destination,
            const std::vector<T> &source,
            const BenchmarkStream &stream)
        {
            if (destination.bytes() != source.size() * sizeof(T))
                throw std::logic_error("device upload capacity mismatch");
            copyHostToDeviceAsync(
                destination.data(), source.data(), destination.bytes(), stream);
        }

        /** Download an entire typed allocation at a correctness boundary. */
        template <typename T>
        [[nodiscard]] std::vector<T> downloadVector(
            const DeviceBuffer<T> &source,
            const BenchmarkStream &stream)
        {
            std::vector<T> result(source.bytes() / sizeof(T));
            copyDeviceToHostAsync(
                result.data(), source.data(), source.bytes(), stream);
            stream.synchronize();
            return result;
        }

        /** Fail with the exact first byte/element mismatch. */
        template <typename T>
        void requireEqual(
            std::span<const T> observed,
            std::span<const T> expected,
            std::string_view region)
        {
            if (observed.size() != expected.size())
                throw std::runtime_error(
                    std::string(region) + " capacity mismatch");
            const auto mismatch =
                std::mismatch(observed.begin(), observed.end(), expected.begin());
            if (mismatch.first == observed.end())
                return;
            const std::size_t index = static_cast<std::size_t>(
                std::distance(observed.begin(), mismatch.first));
            throw std::runtime_error(
                std::string(region) + " mismatch at element " +
                std::to_string(index) + ": observed=" +
                std::to_string(static_cast<std::uint64_t>(*mismatch.first)) +
                " expected=" +
                std::to_string(static_cast<std::uint64_t>(*mismatch.second)));
        }

        /** Runtime-neutral production demotion-kernel dispatch. */
        [[nodiscard]] bool launchGpuToCpu(
            const ExpertTierGpuConstProjectionView &source,
            const ExpertTierWeightDeviceLayout &layout,
            std::uint32_t first_unit,
            std::uint32_t unit_count,
            std::uint8_t *device_chunk,
            std::size_t device_chunk_capacity,
            const BenchmarkStream &stream)
        {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            return launchGpuToCpuExpertTierChunkCUDA(
                source,
                layout,
                first_unit,
                unit_count,
                device_chunk,
                device_chunk_capacity,
                stream.opaque());
#else
            return launchGpuToCpuExpertTierChunkROCm(
                source,
                layout,
                first_unit,
                unit_count,
                device_chunk,
                device_chunk_capacity,
                stream.opaque());
#endif
        }

        /** Runtime-neutral production promotion-kernel dispatch. */
        [[nodiscard]] bool launchCpuToGpu(
            const std::uint8_t *device_chunk,
            std::size_t device_chunk_bytes,
            const ExpertTierWeightDeviceLayout &layout,
            std::uint32_t first_unit,
            std::uint32_t unit_count,
            const ExpertTierGpuMutableProjectionView &destination,
            const BenchmarkStream &stream)
        {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            return launchCpuToGpuExpertTierChunkCUDA(
                device_chunk,
                device_chunk_bytes,
                layout,
                first_unit,
                unit_count,
                destination,
                stream.opaque());
#else
            return launchCpuToGpuExpertTierChunkROCm(
                device_chunk,
                device_chunk_bytes,
                layout,
                first_unit,
                unit_count,
                destination,
                stream.opaque());
#endif
        }

        /**
         * Own every persistent allocation used by one format/shape fixture.
         * Construction ends with all uploads observed on the exact stream.
         */
        class Fixture final
        {
        public:
            /** Prepare source/CPU/destination representations and device storage. */
            Fixture(
                const test::QuantizedVerifierFormatCase &format_case,
                ProjectionShape shape,
                std::uint32_t units_per_chunk,
                int numa_node)
                : format_case_(format_case),
                  shape_(shape),
                  numa_node_(numa_node),
                  tensor_(format_case.create(
                      {static_cast<std::size_t>(shape.N),
                       static_cast<std::size_t>(shape.K)},
                      91001u + format_case.source_codebook_id * 101u +
                          static_cast<std::uint32_t>(shape.N))),
                  source_host_(requireAndPackTensor()),
                  source_format_(requireSourceFormat()),
                  demotion_manifest_(
                      makeGpuToCpuExpertTierWeightStreamManifest(
                          source_format_,
                          shape.N,
                          shape.K,
                          1,
                          0,
                          0,
                          ExpertTierWeightProjection::Gate,
                          checkedChunkUnits(units_per_chunk))),
                  demotion_layout_(demotion_manifest_.deviceLayout()),
                  cpu_host_(packCpuWeights()),
                  promotion_manifest_(
                      makeCpuToGpuExpertTierWeightStreamManifest(
                          cpu_host_,
                          2,
                          0,
                          0,
                          ExpertTierWeightProjection::Gate,
                          checkedChunkUnits(units_per_chunk))),
                  promotion_layout_(promotion_manifest_.deviceLayout()),
                  destination_host_(makeDestinationReference()),
                  source_payload_(source_host_.payload.size()),
                  source_scales_(source_host_.scales.size()),
                  source_mins_(source_host_.mins.size()),
                  source_emins_(source_host_.emins.size()),
                  destination_payload_(destination_host_.payload.size()),
                  destination_scales_(destination_host_.scales.size()),
                  destination_mins_(destination_host_.mins.size()),
                  destination_emins_(destination_host_.emins.size()),
                  device_chunk_(std::max(
                      demotion_layout_.chunkBytes(
                          demotion_layout_.maximum_units_per_chunk),
                      promotion_layout_.chunkBytes(
                          promotion_layout_.maximum_units_per_chunk))),
                  pinned_cpu_(cpu_host_.native_interleaved.size())
            {
                if (!tensor_)
                    throw std::runtime_error("format factory returned null");
                if (!demotion_layout_.valid() || !promotion_layout_.valid())
                    throw std::runtime_error("fixture produced an invalid device layout");
                if (cpu_host_.native_interleaved.size() != pinned_cpu_.size())
                    throw std::runtime_error("CPU packed byte capacity mismatch");

                std::memcpy(
                    pinned_cpu_.data(),
                    cpu_host_.native_interleaved.data(),
                    pinned_cpu_.size());
                uploadVector(source_payload_, source_host_.payload, stream_);
                uploadVector(source_scales_, source_host_.scales, stream_);
                uploadVector(source_mins_, source_host_.mins, stream_);
                uploadVector(source_emins_, source_host_.emins, stream_);
                zeroDeviceAsync(
                    destination_payload_.data(),
                    destination_payload_.bytes(),
                    stream_);
                zeroDeviceAsync(
                    destination_scales_.data(),
                    destination_scales_.bytes(),
                    stream_);
                zeroDeviceAsync(
                    destination_mins_.data(),
                    destination_mins_.bytes(),
                    stream_);
                zeroDeviceAsync(
                    destination_emins_.data(),
                    destination_emins_.bytes(),
                    stream_);
                stream_.synchronize();

                if (!sourceView().validFor(demotion_layout_))
                    throw std::runtime_error("source device view is undersized");
                if (!destinationView().validFor(promotion_layout_))
                    throw std::runtime_error("destination device view is undersized");
            }

            Fixture(const Fixture &) = delete;
            Fixture &operator=(const Fixture &) = delete;

            /** @return Canonical format descriptor owned by the global registry. */
            [[nodiscard]] const test::QuantizedVerifierFormatCase &formatCase() const
            {
                return format_case_;
            }

            /** @return Logical model projection shape. */
            [[nodiscard]] ProjectionShape shape() const noexcept { return shape_; }

            /** @return Physical source codebook consumed by demotion. */
            [[nodiscard]] std::uint8_t sourcePhysicalCodebook() const noexcept
            {
                return source_host_.codebook_id;
            }

            /** @return Physical destination codebook produced by promotion. */
            [[nodiscard]] std::uint8_t destinationPhysicalCodebook() const noexcept
            {
                return destination_host_.codebook_id;
            }

            /** Prove both full streaming directions against their host oracles. */
            void validateRoundTrip()
            {
                enqueueDemotionStream();
                stream_.synchronize();
                requireEqual<std::uint8_t>(
                    std::span<const std::uint8_t>(
                        pinned_cpu_.data(), pinned_cpu_.size()),
                    std::span<const std::uint8_t>(
                        cpu_host_.native_interleaved.data(),
                        cpu_host_.native_interleaved.size()),
                    "GPU-to-CPU stream");

                enqueuePromotionStream();
                stream_.synchronize();
                requireEqual<std::uint8_t>(
                    downloadVector(destination_payload_, stream_),
                    destination_host_.payload,
                    "CPU-to-GPU payload");
                requireEqual<std::uint16_t>(
                    downloadVector(destination_scales_, stream_),
                    destination_host_.scales,
                    "CPU-to-GPU scales");
                requireEqual<std::uint16_t>(
                    downloadVector(destination_mins_, stream_),
                    destination_host_.mins,
                    "CPU-to-GPU mins");
                requireEqual<std::uint32_t>(
                    downloadVector(destination_emins_, stream_),
                    destination_host_.emins,
                    "CPU-to-GPU emins");
            }

            /**
             * Measure one kernel-only chunk or a full kernel+DMA stream.
             *
             * Each sample is a complete event-timed iteration batch. The
             * returned duration is the median batch mean, which rejects
             * transient clock and copy-engine interference while preserving
             * the real asynchronous stream DAG. The caller-provided economy
             * floor is recorded in the artifact alongside the pass/fail bit.
             */
            [[nodiscard]] Measurement measure(
                Direction direction,
                MeasurementMode mode,
                int warmup,
                int iterations,
                int samples,
                double economy_floor_gbps)
            {
                if (iterations <= 0 || warmup < 0 || samples <= 0 ||
                    !std::isfinite(economy_floor_gbps) ||
                    economy_floor_gbps < 0.0)
                    throw std::invalid_argument("invalid timing iteration count");

                // The kernel-only promotion sample excludes staging DMA.  Put
                // its first CPU chunk on device before either warmup or the
                // start event so a zero-warmup measurement is still honest.
                if (mode == MeasurementMode::Kernel &&
                    direction == Direction::CpuToGpu &&
                    !promotion_input_prepared_)
                    preparePromotionKernelInput();

                auto enqueue = [&]
                {
                    if (mode == MeasurementMode::Kernel)
                    {
                        enqueueOneKernel(direction);
                    }
                    else if (direction == Direction::GpuToCpu)
                    {
                        enqueueDemotionStream();
                    }
                    else
                    {
                        enqueuePromotionStream();
                    }
                };

                for (int iteration = 0; iteration < warmup; ++iteration)
                    enqueue();
                stream_.synchronize();

                // Each batch uses independent events, but all events and
                // storage exist outside their measured intervals. Taking the
                // median rejects one power-state or copy-engine scheduling
                // outlier without selecting the unrealistically fastest run.
                std::vector<double> sample_mean_ms;
                sample_mean_ms.reserve(static_cast<std::size_t>(samples));
                for (int sample = 0; sample < samples; ++sample)
                {
                    BenchmarkEvent begin;
                    BenchmarkEvent end;
                    begin.record(stream_);
                    for (int iteration = 0; iteration < iterations; ++iteration)
                        enqueue();
                    end.record(stream_);
                    end.synchronize();
                    sample_mean_ms.push_back(
                        BenchmarkEvent::elapsed(begin, end) /
                        static_cast<double>(iterations));
                }
                std::sort(sample_mean_ms.begin(), sample_mean_ms.end());
                const double mean_ms = sample_mean_ms[
                    sample_mean_ms.size() / 2u];
                const auto &layout = direction == Direction::GpuToCpu
                                         ? demotion_layout_
                                         : promotion_layout_;
                const std::uint32_t chunk_units =
                    layout.maximum_units_per_chunk;
                const std::uint64_t bytes =
                    mode == MeasurementMode::Kernel
                        ? layout.chunkBytes(chunk_units)
                        : static_cast<std::uint64_t>(
                              direction == Direction::GpuToCpu
                                  ? demotion_manifest_.total_stream_bytes
                                  : promotion_manifest_.total_stream_bytes);
                const std::uint32_t chunks =
                    mode == MeasurementMode::Kernel
                        ? 1u
                        : (layout.unit_count + chunk_units - 1u) / chunk_units;

                const double gb_per_s = mean_ms > 0.0
                                            ? static_cast<double>(bytes) /
                                                  (mean_ms * 1.0e6)
                                            : 0.0;
                return Measurement{
                    .backend = kBackendName,
                    .numa_node = numa_node_,
                    .mode = mode,
                    .direction = direction,
                    .format = format_case_.label,
                    .source_codebook = format_case_.source_codebook_id,
                    .physical_codebook =
                        direction == Direction::GpuToCpu
                            ? sourcePhysicalCodebook()
                            : destinationPhysicalCodebook(),
                    .shape = shape_,
                    .unit_count = layout.unit_count,
                    .units_per_chunk = chunk_units,
                    .chunk_count = chunks,
                    .semantic_bytes = bytes,
                    .mean_ms = mean_ms,
                    .gb_per_s = gb_per_s,
                    .iterations = iterations,
                    .samples = samples,
                    .economy_floor_gbps = economy_floor_gbps,
                    .economical = gb_per_s >= economy_floor_gbps,
                };
            }

            /** Launch exactly one selected kernel for an isolated profiler run. */
            void singleLaunch(Direction direction)
            {
                if (direction == Direction::CpuToGpu &&
                    !promotion_input_prepared_)
                    preparePromotionKernelInput();
                enqueueOneKernel(direction);
                stream_.synchronize();
            }

        private:
            /** Validate the newly created tensor before member initialization proceeds. */
            [[nodiscard]] HostGpuExpertPackedProjection requireAndPackTensor() const
            {
                if (!tensor_)
                    throw std::runtime_error("format factory returned null");
                return packGpuProjection(*tensor_);
            }

            /** Resolve exact source provenance; aliases require superblock identity. */
            [[nodiscard]] const NativeVnniFormatInfo &requireSourceFormat() const
            {
                const NativeVnniFormatInfo *format =
                    native_vnni_formats::forSourceIdentity(
                        source_host_.source_codebook_id,
                        source_host_.is_superblock);
                if (format == nullptr)
                    throw std::runtime_error("source format is absent from the catalog");
                return *format;
            }

            /** Limit requested staging capacity to this shape's complete units. */
            [[nodiscard]] std::uint32_t checkedChunkUnits(
                std::uint32_t requested) const
            {
                if (requested == 0)
                    throw std::invalid_argument("units-per-chunk must be positive");
                const std::uint64_t units =
                    static_cast<std::uint64_t>((shape_.N + 63) / 64) *
                    static_cast<std::uint64_t>(shape_.K / 32);
                if (units == 0 ||
                    units > std::numeric_limits<std::uint32_t>::max())
                    throw std::invalid_argument("projection unit count is invalid");
                return std::min<std::uint32_t>(
                    requested, static_cast<std::uint32_t>(units));
            }

            /** Build the byte-exact final CPU representation used on the wire. */
            [[nodiscard]] cpu::native_vnni::CPUNativeVNNIPackedWeights
            packCpuWeights() const
            {
                cpu::native_vnni::CPUNativeVNNIPackedWeights packed;
                if (!cpu::native_vnni::packWeightsCPUNativeVNNI(
                        tensor_.get(), packed))
                    throw std::runtime_error("production CPU weight pack failed");
                if (packed.native_interleaved.size() !=
                    demotion_manifest_.total_stream_bytes)
                    throw std::runtime_error(
                        "CPU oracle disagrees with demotion manifest byte count");
                return packed;
            }

            /** Build the normalized accelerator representation expected after promotion. */
            [[nodiscard]] HostGpuExpertPackedProjection
            makeDestinationReference() const
            {
                HostGpuExpertPackedProjection result;
                std::string error;
                if (!cpuToGpuExpertPackedReference(cpu_host_, result, &error))
                    throw std::runtime_error(
                        "CPU-to-GPU reference conversion failed: " + error);
                return result;
            }

            /** @return Exact read-only view over persistent source allocations. */
            [[nodiscard]] ExpertTierGpuConstProjectionView sourceView() const
            {
                return ExpertTierGpuConstProjectionView{
                    .payload = source_payload_.data(),
                    .scales = source_scales_.data(),
                    .mins = source_mins_.data(),
                    .emins = source_emins_.data(),
                    .payload_bytes = source_payload_.bytes(),
                    .scales_bytes = source_scales_.bytes(),
                    .mins_bytes = source_mins_.bytes(),
                    .emins_bytes = source_emins_.bytes(),
                };
            }

            /** @return Exact writable view over persistent destination allocations. */
            [[nodiscard]] ExpertTierGpuMutableProjectionView destinationView()
            {
                return ExpertTierGpuMutableProjectionView{
                    .payload = destination_payload_.data(),
                    .scales = destination_scales_.data(),
                    .mins = destination_mins_.data(),
                    .emins = destination_emins_.data(),
                    .payload_bytes = destination_payload_.bytes(),
                    .scales_bytes = destination_scales_.bytes(),
                    .mins_bytes = destination_mins_.bytes(),
                    .emins_bytes = destination_emins_.bytes(),
                };
            }

            /** Enqueue one chunk conversion without any DMA for kernel profiling. */
            void enqueueOneKernel(Direction direction)
            {
                const auto &layout = direction == Direction::GpuToCpu
                                         ? demotion_layout_
                                         : promotion_layout_;
                const std::uint32_t units = layout.maximum_units_per_chunk;
                const std::size_t bytes = layout.chunkBytes(units);
                if (direction == Direction::GpuToCpu)
                {
                    if (!launchGpuToCpu(
                            sourceView(),
                            layout,
                            0,
                            units,
                            device_chunk_.data(),
                            device_chunk_.bytes(),
                            stream_))
                        throw std::runtime_error("GPU-to-CPU kernel launch rejected");
                    return;
                }

                // Promotion's staging input is populated outside the measured
                // kernel in `preparePromotionKernelInput`.
                if (!promotion_input_prepared_)
                    preparePromotionKernelInput();
                if (!launchCpuToGpu(
                        device_chunk_.data(),
                        bytes,
                        layout,
                        0,
                        units,
                        destinationView(),
                        stream_))
                    throw std::runtime_error("CPU-to-GPU kernel launch rejected");
            }

            /** Upload the first promotion chunk once, before kernel-only timing. */
            void preparePromotionKernelInput()
            {
                const std::size_t bytes = promotion_layout_.chunkBytes(
                    promotion_layout_.maximum_units_per_chunk);
                copyHostToDeviceAsync(
                    device_chunk_.data(), pinned_cpu_.data(), bytes, stream_);
                stream_.synchronize();
                promotion_input_prepared_ = true;
            }

            /** Enqueue a full device-repack plus pinned D2H projection stream. */
            void enqueueDemotionStream()
            {
                const std::uint32_t chunk_capacity =
                    demotion_layout_.maximum_units_per_chunk;
                for (std::uint32_t first = 0;
                     first < demotion_layout_.unit_count;
                     first += chunk_capacity)
                {
                    const std::uint32_t units = std::min<std::uint32_t>(
                        chunk_capacity, demotion_layout_.unit_count - first);
                    const std::size_t bytes = demotion_layout_.chunkBytes(units);
                    if (!launchGpuToCpu(
                            sourceView(),
                            demotion_layout_,
                            first,
                            units,
                            device_chunk_.data(),
                            device_chunk_.bytes(),
                            stream_))
                        throw std::runtime_error(
                            "GPU-to-CPU streaming kernel launch rejected");
                    copyDeviceToHostAsync(
                        pinned_cpu_.data() + demotion_layout_.chunkBytes(first),
                        device_chunk_.data(),
                        bytes,
                        stream_);
                }
            }

            /** Enqueue pinned H2D chunks plus direct final-device repacking. */
            void enqueuePromotionStream()
            {
                const std::uint32_t chunk_capacity =
                    promotion_layout_.maximum_units_per_chunk;
                for (std::uint32_t first = 0;
                     first < promotion_layout_.unit_count;
                     first += chunk_capacity)
                {
                    const std::uint32_t units = std::min<std::uint32_t>(
                        chunk_capacity, promotion_layout_.unit_count - first);
                    const std::size_t bytes = promotion_layout_.chunkBytes(units);
                    copyHostToDeviceAsync(
                        device_chunk_.data(),
                        pinned_cpu_.data() + promotion_layout_.chunkBytes(first),
                        bytes,
                        stream_);
                    if (!launchCpuToGpu(
                            device_chunk_.data(),
                            bytes,
                            promotion_layout_,
                            first,
                            units,
                            destinationView(),
                            stream_))
                        throw std::runtime_error(
                            "CPU-to-GPU streaming kernel launch rejected");
                }
            }

            const test::QuantizedVerifierFormatCase &format_case_;
            ProjectionShape shape_;
            int numa_node_ = -1;
            std::unique_ptr<TensorBase> tensor_;
            HostGpuExpertPackedProjection source_host_;
            const NativeVnniFormatInfo &source_format_;
            ExpertTierWeightStreamManifest demotion_manifest_;
            ExpertTierWeightDeviceLayout demotion_layout_;
            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu_host_;
            ExpertTierWeightStreamManifest promotion_manifest_;
            ExpertTierWeightDeviceLayout promotion_layout_;
            HostGpuExpertPackedProjection destination_host_;
            BenchmarkStream stream_;
            DeviceBuffer<std::uint8_t> source_payload_;
            DeviceBuffer<std::uint16_t> source_scales_;
            DeviceBuffer<std::uint16_t> source_mins_;
            DeviceBuffer<std::uint32_t> source_emins_;
            DeviceBuffer<std::uint8_t> destination_payload_;
            DeviceBuffer<std::uint16_t> destination_scales_;
            DeviceBuffer<std::uint16_t> destination_mins_;
            DeviceBuffer<std::uint32_t> destination_emins_;
            DeviceBuffer<std::uint8_t> device_chunk_;
            PinnedBytes pinned_cpu_;
            bool promotion_input_prepared_ = false;
        };

        /** Parse a bounded non-negative integer without locale-dependent APIs. */
        [[nodiscard]] int parseInteger(
            std::string_view text,
            std::string_view option,
            int minimum,
            int maximum)
        {
            int value = 0;
            const auto result = std::from_chars(
                text.data(), text.data() + text.size(), value);
            if (result.ec != std::errc{} ||
                result.ptr != text.data() + text.size() ||
                value < minimum || value > maximum)
                throw std::invalid_argument(
                    std::string(option) + " has an invalid integer value");
            return value;
        }

        /** Parse a finite non-negative throughput floor. */
        [[nodiscard]] double parseThroughput(
            std::string_view text,
            std::string_view option)
        {
            double value = 0.0;
            const auto result = std::from_chars(
                text.data(), text.data() + text.size(), value);
            if (result.ec != std::errc{} ||
                result.ptr != text.data() + text.size() ||
                !std::isfinite(value) || value < 0.0)
                throw std::invalid_argument(
                    std::string(option) +
                    " has an invalid throughput value");
            return value;
        }

        /** Print stable CLI help used by profiler scripts and humans. */
        void printUsage(std::ostream &output, std::string_view program)
        {
            output
                << "Usage: " << program << " [options]\n"
                << "  --format NAME|all          Quantized source format (default all)\n"
                << "  --shape gate_up|down|all   Real expert projection shape\n"
                << "  --direction gpu-to-cpu|cpu-to-gpu|both\n"
                << "  --mode kernel|streaming|both\n"
                << "  --units-per-chunk N        Complete 64-column CPU units (default 512)\n"
                << "  --warmup N                 Unmeasured iterations (default 5)\n"
                << "  --iterations N             Iterations per sample (default 20)\n"
                << "  --samples N                Median timing batches (default 3)\n"
                << "  --device N                 Backend device ordinal (default 0)\n"
                << "  --numa auto|none|N         Host affinity for pinned DMA (default auto)\n"
                << "  --minimum-kernel-gbps N    Per-row economy floor (default 5)\n"
                << "  --minimum-streaming-gbps N Per-row economy floor (default 1)\n"
                << "  --csv PATH                 Also write measurement CSV to PATH\n"
                << "  --single-launch            One kernel only; requires singular axes\n";
        }

        /** Parse long options in either `--name value` or `--name=value` form. */
        [[nodiscard]] Options parseOptions(int argc, char **argv)
        {
            Options options;
            for (int index = 1; index < argc; ++index)
            {
                std::string_view argument(argv[index]);
                if (argument == "--help" || argument == "-h")
                {
                    printUsage(std::cout, argv[0]);
                    std::exit(0);
                }
                if (argument == "--single-launch")
                {
                    options.single_launch = true;
                    continue;
                }
                if (!argument.starts_with("--"))
                    throw std::invalid_argument(
                        "unexpected positional argument: " +
                        std::string(argument));

                const std::size_t equals = argument.find('=');
                const std::string_view name = argument.substr(0, equals);
                std::string_view value;
                if (equals != std::string_view::npos)
                {
                    value = argument.substr(equals + 1);
                }
                else
                {
                    if (++index >= argc)
                        throw std::invalid_argument(
                            std::string(name) + " requires a value");
                    value = argv[index];
                }

                if (name == "--format")
                    options.format = value;
                else if (name == "--shape")
                    options.shape = value;
                else if (name == "--direction")
                    options.direction = value;
                else if (name == "--mode")
                    options.mode = value;
                else if (name == "--units-per-chunk")
                    options.units_per_chunk = static_cast<std::uint32_t>(
                        parseInteger(value, name, 1, 1 << 20));
                else if (name == "--warmup")
                    options.warmup = parseInteger(value, name, 0, 1000000);
                else if (name == "--iterations")
                    options.iterations = parseInteger(value, name, 1, 1000000);
                else if (name == "--samples")
                    options.samples = parseInteger(value, name, 1, 101);
                else if (name == "--device")
                    options.device = parseInteger(value, name, 0, 1024);
                else if (name == "--numa")
                    options.numa = value;
                else if (name == "--minimum-kernel-gbps")
                    options.minimum_kernel_gbps =
                        parseThroughput(value, name);
                else if (name == "--minimum-streaming-gbps")
                    options.minimum_streaming_gbps =
                        parseThroughput(value, name);
                else if (name == "--csv")
                    options.csv_path = std::string(value);
                else
                    throw std::invalid_argument(
                        "unknown option: " + std::string(name));
            }
            if ((options.samples % 2) == 0)
                throw std::invalid_argument(
                    "--samples must be odd so the median is unambiguous");
            return options;
        }

        /**
         * Bind host execution to the selected GPU's NUMA node before pinned
         * memory is allocated and first-touched.
         *
         * PCI sysfs is the authority rather than device ordinal, MPI rank, or
         * backend name. Moving a card between sockets therefore changes the
         * benchmark placement without changing a test topology declaration.
         */
        [[nodiscard]] int bindHostNuma(const Options &options)
        {
            if (options.numa == "none")
                return -1;

            int node = -1;
            if (options.numa == "auto")
            {
                const std::string pci_bus = devicePciBusId(options.device);
                const std::string path =
                    "/sys/bus/pci/devices/" + pci_bus + "/numa_node";
                std::ifstream input(path);
                if (!input || !(input >> node))
                    throw std::runtime_error(
                        "cannot resolve GPU NUMA node from " + path);
            }
            else
            {
                node = parseInteger(
                    options.numa,
                    "--numa",
                    0,
                    std::numeric_limits<int>::max());
            }

            // A negative sysfs node means the platform exposes no locality.
            // Retaining the caller's affinity is the only honest behavior.
            if (node < 0)
                return -1;
            if (numa_available() < 0 || node > numa_max_node())
                throw std::runtime_error(
                    "selected GPU reports an unavailable NUMA node " +
                    std::to_string(node));

            errno = 0;
            if (numa_run_on_node(node) != 0)
                throw std::runtime_error(
                    "cannot bind benchmark to NUMA node " +
                    std::to_string(node) + ": " + std::strerror(errno));
            return node;
        }

        /** Resolve selected format entries while retaining registry ownership. */
        [[nodiscard]] std::vector<const test::QuantizedVerifierFormatCase *>
        selectedFormats(const Options &options)
        {
            std::vector<const test::QuantizedVerifierFormatCase *> result;
            for (const auto &format : test::quantizedVerifierFormats())
            {
                if (options.format == "all" || options.format == format.label)
                    result.push_back(&format);
            }
            if (result.empty())
                throw std::invalid_argument(
                    "unknown quantized format: " + options.format);
            return result;
        }

        /** Resolve the selected real expert projection geometries. */
        [[nodiscard]] std::vector<ProjectionShape> selectedShapes(
            const Options &options)
        {
            std::vector<ProjectionShape> result;
            if (options.shape == "all" || options.shape == kGateUpShape.name)
                result.push_back(kGateUpShape);
            if (options.shape == "all" || options.shape == kDownShape.name)
                result.push_back(kDownShape);
            if (result.empty())
                throw std::invalid_argument(
                    "unknown projection shape: " + options.shape);
            return result;
        }

        /** Resolve one or both conversion directions. */
        [[nodiscard]] std::vector<Direction> selectedDirections(
            const Options &options)
        {
            std::vector<Direction> result;
            if (options.direction == "both" ||
                options.direction == "gpu-to-cpu")
                result.push_back(Direction::GpuToCpu);
            if (options.direction == "both" ||
                options.direction == "cpu-to-gpu")
                result.push_back(Direction::CpuToGpu);
            if (result.empty())
                throw std::invalid_argument(
                    "unknown transfer direction: " + options.direction);
            return result;
        }

        /** Resolve kernel-only, end-to-end streaming, or both measurements. */
        [[nodiscard]] std::vector<MeasurementMode> selectedModes(
            const Options &options)
        {
            std::vector<MeasurementMode> result;
            if (options.mode == "both" || options.mode == "kernel")
                result.push_back(MeasurementMode::Kernel);
            if (options.mode == "both" || options.mode == "streaming")
                result.push_back(MeasurementMode::Streaming);
            if (result.empty())
                throw std::invalid_argument(
                    "unknown measurement mode: " + options.mode);
            return result;
        }

        /** Stable textual direction for CSV evidence and profiler filenames. */
        [[nodiscard]] std::string_view directionName(Direction direction)
        {
            return direction == Direction::GpuToCpu
                       ? "gpu_to_cpu"
                       : "cpu_to_gpu";
        }

        /** Stable textual scope for CSV evidence. */
        [[nodiscard]] std::string_view modeName(MeasurementMode mode)
        {
            return mode == MeasurementMode::Kernel ? "kernel" : "streaming";
        }

        /** Emit the canonical header shared by stdout and optional artifacts. */
        void writeCsvHeader(std::ostream &output)
        {
            output
                << "backend,numa_node,mode,direction,format,source_codebook,physical_codebook,"
                   "shape,n,k,unit_count,units_per_chunk,chunk_count,semantic_bytes,"
                   "mean_ms,gb_per_s,iterations,samples,economy_floor_gbps,"
                   "validated,economical\n";
        }

        /** Emit one fully reproducible measurement row. */
        void writeCsvRow(std::ostream &output, const Measurement &measurement)
        {
            output << measurement.backend << ','
                   << measurement.numa_node << ','
                   << modeName(measurement.mode) << ','
                   << directionName(measurement.direction) << ','
                   << measurement.format << ','
                   << static_cast<unsigned>(measurement.source_codebook) << ','
                   << static_cast<unsigned>(measurement.physical_codebook) << ','
                   << measurement.shape.name << ','
                   << measurement.shape.N << ','
                   << measurement.shape.K << ','
                   << measurement.unit_count << ','
                   << measurement.units_per_chunk << ','
                   << measurement.chunk_count << ','
                   << measurement.semantic_bytes << ','
                   << std::fixed << std::setprecision(6)
                   << measurement.mean_ms << ','
                   << measurement.gb_per_s << ','
                   << measurement.iterations << ','
                   << measurement.samples << ','
                   << measurement.economy_floor_gbps << ",true,"
                   << (measurement.economical ? "true" : "false") << '\n';
        }

        /** Initialize backend-local IQ tables before any conversion is launched. */
        void initializeDecodeTables(int device)
        {
#if defined(LLAMINAR_EXPERT_TIER_PERF_CUDA)
            if (!cudaNativeVNNIInitIQGridTables_tuned())
                throw std::runtime_error(
                    "CUDA production IQ table initialization failed");
#else
            if (!rocm::ensureIQGridTablesInitialized(device))
                throw std::runtime_error(
                    "ROCm production IQ table initialization failed");
#endif
        }

        /** Run the selected matrix and return process status through exceptions. */
        int run(int argc, char **argv)
        {
            const Options options = parseOptions(argc, argv);
            const auto formats = selectedFormats(options);
            const auto shapes = selectedShapes(options);
            const auto directions = selectedDirections(options);
            const auto modes = selectedModes(options);

            if (options.device >= deviceCount())
                throw std::invalid_argument(
                    "requested backend device is not available");
            selectDevice(options.device);
            const int numa_node = bindHostNuma(options);
            initializeDecodeTables(options.device);
            std::cerr << "ExpertTierWeightStreaming backend=" << kBackendName
                      << " device=" << options.device
                      << " numa_node=" << numa_node << '\n';

            if (options.single_launch &&
                (formats.size() != 1 || shapes.size() != 1 ||
                 directions.size() != 1))
                throw std::invalid_argument(
                    "--single-launch requires one format, shape, and direction");

            std::optional<std::ofstream> csv_file;
            if (options.csv_path.has_value())
            {
                csv_file.emplace(*options.csv_path, std::ios::trunc);
                if (!*csv_file)
                    throw std::runtime_error(
                        "cannot open CSV output: " + *options.csv_path);
            }

            if (options.single_launch)
            {
                Fixture fixture(
                    *formats.front(),
                    shapes.front(),
                    options.units_per_chunk,
                    numa_node);
                fixture.singleLaunch(directions.front());
                std::cout << "single_launch_backend=" << kBackendName
                          << " numa_node=" << numa_node
                          << " format=" << formats.front()->label
                          << " shape=" << shapes.front().name
                          << " direction=" << directionName(directions.front())
                          << " units="
                          << std::min<std::uint32_t>(
                                 options.units_per_chunk,
                                 static_cast<std::uint32_t>(
                                     ((shapes.front().N + 63) / 64) *
                                     (shapes.front().K / 32)))
                          << '\n';
                return 0;
            }

            writeCsvHeader(std::cout);
            if (csv_file)
                writeCsvHeader(*csv_file);

            std::size_t uneconomical_rows = 0;
            for (const auto *format : formats)
            {
                for (ProjectionShape shape : shapes)
                {
                    Fixture fixture(
                        *format,
                        shape,
                        options.units_per_chunk,
                        numa_node);
                    fixture.validateRoundTrip();
                    for (Direction direction : directions)
                    {
                        for (MeasurementMode mode : modes)
                        {
                            const double economy_floor =
                                mode == MeasurementMode::Kernel
                                    ? options.minimum_kernel_gbps
                                    : options.minimum_streaming_gbps;
                            const Measurement measurement = fixture.measure(
                                direction,
                                mode,
                                options.warmup,
                                options.iterations,
                                options.samples,
                                economy_floor);
                            writeCsvRow(std::cout, measurement);
                            if (csv_file)
                                writeCsvRow(*csv_file, measurement);
                            if (!measurement.economical)
                                ++uneconomical_rows;
                        }
                    }
                }
            }
            std::cout.flush();
            if (csv_file)
                csv_file->flush();
            if (uneconomical_rows != 0)
                throw std::runtime_error(
                    std::to_string(uneconomical_rows) +
                    " validated tier conversion rows missed their economy floor");
            return 0;
        }
    } // namespace
} // namespace llaminar2

/** Standalone Release benchmark entry point. */
int main(int argc, char **argv)
{
    try
    {
        return llaminar2::run(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "ExpertTierWeightStreaming benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
