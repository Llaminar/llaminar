/**
 * @file CrossVendorNativeVNNIArithmeticEmitter.cpp
 * @brief Emit exact CUDA or ROCm grouped-MoE evidence for every codebook.
 *
 * CUDA and HIP device runtimes cannot safely register every linked device
 * image in one diagnostic process on all supported driver combinations. The
 * cross-vendor gate therefore builds this source twice and launches one
 * backend and one source format per process. Each invocation prepares weights
 * through the production loader, executes top-8 decode and grouped prefill at
 * Qwen3.5-122B expert geometry, and retains exact CSV and binary checkpoints.
 * The companion Python driver joins those process-isolated artifacts without
 * introducing a numerical tolerance.
 */

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/TensorKernels.h"

#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kDModel = 3072;
        constexpr int kIntermediate = 1024;
        constexpr int kNumExperts = 8;
        constexpr int kTopK = 8;
        constexpr int kWeightVariants = 2;
        constexpr std::array<int, 2> kGroupedRows = {9, 16};

        /** @brief Own the one exact non-default stream used by an emitter. */
        class ExplicitStream final
        {
        public:
            /**
             * @brief Create an explicit stream on @p device.
             * @throws std::invalid_argument for a non-GPU backend.
             * @throws std::runtime_error when stream creation fails.
             */
            ExplicitStream(IBackend *backend, DeviceId device)
                : backend_(backend), device_(device)
            {
                if (!backend_ || !device_.is_gpu())
                {
                    throw std::invalid_argument(
                        "Cross-vendor arithmetic evidence requires a GPU backend");
                }
                stream_ = backend_->createStream(device_.ordinal);
                if (!stream_)
                {
                    throw std::runtime_error(
                        "Cross-vendor arithmetic evidence could not create a stream");
                }
            }

            /** @brief Destroy the exact stream after every dependent owner. */
            ~ExplicitStream()
            {
                if (stream_)
                    backend_->destroyStream(stream_, device_.ordinal);
            }

            ExplicitStream(const ExplicitStream &) = delete;
            ExplicitStream &operator=(const ExplicitStream &) = delete;

            /** @return The exact non-null stream used by every GPU operation. */
            [[nodiscard]] void *get() const noexcept { return stream_; }

        private:
            IBackend *backend_ = nullptr; ///< Non-owning backend singleton.
            DeviceId device_{};           ///< Device on which the stream lives.
            void *stream_ = nullptr;      ///< Backend-owned explicit stream.
        };

        /** @brief Lifetime bundle for one production-prepared matrix. */
        struct PreparedMatrix
        {
            std::unique_ptr<TensorBase> source; ///< Native source-format bytes.
            GpuPreparedGemm prepared;           ///< Stable prepared-weight owner.
            DeviceNativeVNNIMatrixDesc descriptor{}; ///< Exported device ABI.
        };

        /** @brief One gate/up/down weight variant reused by alternating experts. */
        struct PreparedExpert
        {
            PreparedMatrix gate; ///< Gate projection at [1024, 3072].
            PreparedMatrix up;   ///< Up projection at [1024, 3072].
            PreparedMatrix down; ///< Down projection at [3072, 1024].
        };

        /**
         * @brief Write exact values and binary boundaries for one invocation.
         *
         * Checkpoint and artifact manifests are code-owned descriptions of the
         * evidence. The comparator consumes those manifests rather than
         * duplicating codebook-dependent byte counts in Python.
         */
        class EvidenceWriter final
        {
        public:
            /**
             * @brief Open all CSV destinations beside @p output_path.
             * @throws std::runtime_error when any evidence file cannot open.
             */
            EvidenceWriter(
                IBackend *backend,
                DeviceId device,
                void *stream,
                std::string backend_label,
                std::string format_label,
                std::filesystem::path output_path)
                : backend_(backend),
                  device_(device),
                  stream_(stream),
                  backend_label_(std::move(backend_label)),
                  format_label_(std::move(format_label)),
                  output_path_(std::move(output_path)),
                  checkpoints_(output_path_, std::ios::trunc),
                  checkpoint_manifest_(
                      output_path_.string() + ".checkpoint_manifest.csv",
                      std::ios::trunc),
                  artifact_manifest_(
                      output_path_.string() + ".artifact_manifest.csv",
                      std::ios::trunc),
                  descriptor_manifest_(
                      output_path_.string() + ".descriptor_manifest.csv",
                      std::ios::trunc)
            {
                if (!backend_ || !stream_ || !checkpoints_.is_open() ||
                    !checkpoint_manifest_.is_open() ||
                    !artifact_manifest_.is_open() ||
                    !descriptor_manifest_.is_open())
                {
                    throw std::runtime_error(
                        "Could not open cross-vendor arithmetic evidence");
                }
                checkpoints_ <<
                    "backend,format,case,stage,index,value,bits\n";
                checkpoint_manifest_ <<
                    "backend,format,case,stage,elements\n";
                artifact_manifest_ <<
                    "backend,format,artifact,bytes,element_type,rows,routes,width\n";
                descriptor_manifest_ <<
                    "backend,format,variant,projection,n,k,blocks_per_row,"
                    "execution_codebook,source_codebook,source_superblock,"
                    "payload_bytes_per_block,has_mins,has_emins\n";
            }

            /**
             * @brief Append every exact FP32 word in one host-current tensor.
             */
            void emitTensor(
                std::string_view case_label,
                std::string_view stage,
                const TensorBase &tensor)
            {
                const float *values = tensor.data();
                if (!values || tensor.numel() == 0u)
                    throw std::runtime_error("Arithmetic checkpoint tensor is empty");
                checkpoint_manifest_
                    << backend_label_ << ',' << format_label_ << ','
                    << case_label << ',' << stage << ',' << tensor.numel()
                    << '\n';
                for (size_t index = 0; index < tensor.numel(); ++index)
                {
                    uint32_t bits = 0u;
                    std::memcpy(&bits, values + index, sizeof(bits));
                    checkpoints_
                        << backend_label_ << ',' << format_label_ << ','
                        << case_label << ',' << stage << ',' << index << ','
                        << std::setprecision(
                               std::numeric_limits<float>::max_digits10)
                        << values[index] << ",0x"
                        << std::hex << std::setw(8) << std::setfill('0')
                        << bits << std::dec << std::setfill(' ') << '\n';
                }
            }

            /**
             * @brief Persist an exact host byte span and declare its shape.
             * @param artifact Stable sidecar name appended to the CSV path.
             * @param host_bytes Complete host-current byte span.
             * @param byte_count Number of live bytes at @p host_bytes.
             * @param element_type Stable scalar interpretation for diagnostics.
             * @param rows Optional semantic row count.
             * @param routes Optional routes per semantic row.
             * @param width Optional scalar width per route.
             */
            void emitHostBlob(
                std::string_view artifact,
                const void *host_bytes,
                size_t byte_count,
                std::string_view element_type,
                int rows = 0,
                int routes = 0,
                int width = 0)
            {
                if (!host_bytes || byte_count == 0u)
                    throw std::invalid_argument(
                        "Incomplete host blob " + std::string(artifact));
                const std::filesystem::path path(
                    output_path_.string() + "." + std::string(artifact));
                std::ofstream output(
                    path, std::ios::binary | std::ios::trunc);
                if (!output.is_open())
                    throw std::runtime_error(
                        "Could not open host blob " + path.string());
                output.write(
                    reinterpret_cast<const char *>(host_bytes),
                    static_cast<std::streamsize>(byte_count));
                output.flush();
                if (!output)
                    throw std::runtime_error(
                        "Could not finalize host blob " + path.string());

                artifact_manifest_
                    << backend_label_ << ',' << format_label_ << ','
                    << artifact << ',' << byte_count << ',' << element_type
                    << ',' << rows << ',' << routes << ',' << width << '\n';
            }

            /**
             * @brief Persist an exact device byte span and declare its shape.
             *
             * Shape fields make an FP32 mismatch attributable to a row, route,
             * and column. Prepared-weight artifacts leave those fields zero.
             */
            void emitDeviceBlob(
                std::string_view artifact,
                const void *device_bytes,
                size_t byte_count,
                std::string_view element_type,
                int rows = 0,
                int routes = 0,
                int width = 0)
            {
                if (!device_bytes || byte_count == 0u)
                    throw std::invalid_argument(
                        "Incomplete device blob " + std::string(artifact));
                std::vector<std::byte> host_bytes(byte_count);
                if (!backend_->deviceToHostFast(
                        host_bytes.data(), device_bytes, byte_count,
                        device_.ordinal, stream_) ||
                    !backend_->synchronizeStream(stream_, device_.ordinal))
                {
                    throw std::runtime_error(
                        "Could not observe device blob " +
                        std::string(artifact));
                }

                emitHostBlob(
                    artifact, host_bytes.data(), host_bytes.size(),
                    element_type, rows, routes, width);
            }

            /** @brief Record the complete prepared descriptor contract. */
            void emitDescriptor(
                int variant,
                std::string_view projection,
                const DeviceNativeVNNIMatrixDesc &descriptor,
                const NativeVnniFormatInfo &source_format)
            {
                descriptor_manifest_
                    << backend_label_ << ',' << format_label_ << ','
                    << variant << ',' << projection << ','
                    << descriptor.n << ',' << descriptor.k << ','
                    << descriptor.blocks_per_row << ','
                    << static_cast<unsigned int>(descriptor.codebook_id) << ','
                    << static_cast<unsigned int>(descriptor.source_codebook_id)
                    << ',' << static_cast<unsigned int>(
                           descriptor.source_is_superblock)
                    << ',' << source_format.payload_bytes << ','
                    << (source_format.is_asymmetric ? 1 : 0) << ','
                    << (source_format.has_emins ? 1 : 0) << '\n';
            }

            /** @brief Flush and validate every CSV evidence stream. */
            void finalize()
            {
                checkpoints_.flush();
                checkpoint_manifest_.flush();
                artifact_manifest_.flush();
                descriptor_manifest_.flush();
                if (!checkpoints_ || !checkpoint_manifest_ ||
                    !artifact_manifest_ || !descriptor_manifest_)
                {
                    throw std::runtime_error(
                        "Could not finalize cross-vendor arithmetic evidence");
                }
            }

        private:
            IBackend *backend_ = nullptr; ///< Owner of observed device bytes.
            DeviceId device_{};           ///< Physical evidence device.
            void *stream_ = nullptr;      ///< Exact producer/observer stream.
            std::string backend_label_;   ///< Stable `cuda` or `rocm` label.
            std::string format_label_;    ///< Canonical source-format label.
            std::filesystem::path output_path_; ///< Main checkpoint path.
            std::ofstream checkpoints_;         ///< Exact FP32 word table.
            std::ofstream checkpoint_manifest_; ///< Declared checkpoint sizes.
            std::ofstream artifact_manifest_;   ///< Binary artifact contract.
            std::ofstream descriptor_manifest_; ///< Prepared descriptor ABI.
        };

        /** @brief Resolve the registry ordinal used for stable identities. */
        size_t formatOrdinal(std::string_view label)
        {
            const auto &formats = quantizedMoEVerifierFormats();
            for (size_t index = 0; index < formats.size(); ++index)
            {
                if (label == formats[index].label)
                    return index;
            }
            throw std::invalid_argument(
                "Unknown canonical MoE verifier format: " +
                std::string(label));
        }

        /** @brief Prepare one matrix through the production GPU loader. */
        PreparedMatrix prepareMatrix(
            const QuantizedVerifierFormatCase &format,
            DeviceId device,
            void *stream,
            int rows,
            int columns,
            uint32_t seed,
            std::string name,
            ModelContextId model_id)
        {
            PreparedMatrix result;
            result.source = format.create(
                {static_cast<size_t>(rows), static_cast<size_t>(columns)}, seed);
            result.prepared = makeGpuPreparedGemm(
                result.source.get(), device, name, model_id);
            if (!result.prepared.kernel)
                throw std::runtime_error(name + " produced no GEMM handle");
            if (auto *tensor_kernel =
                    dynamic_cast<ITensorKernel *>(result.prepared.kernel))
            {
                tensor_kernel->setGPUStream(stream);
            }
            if (!result.prepared.kernel->exportNativeVNNIMatrixDesc(
                    result.descriptor))
            {
                throw std::runtime_error(name + " did not export NativeVNNI");
            }

            const auto &descriptor = result.descriptor;
            if (!descriptor.valid() || descriptor.n != rows ||
                descriptor.k != columns ||
                descriptor.codebook_id !=
                    format.device_execution_codebook_id ||
                descriptor.source_identity_present == 0u ||
                descriptor.source_codebook_id != format.source_codebook_id ||
                descriptor.source_is_superblock !=
                    static_cast<uint8_t>(format.source_is_superblock))
            {
                throw std::runtime_error(
                    name + " exported an invalid descriptor contract");
            }
            return result;
        }

        /** @brief Copy a tensor to host after its exact producer stream. */
        void observeTensor(
            IBackend *backend,
            DeviceId device,
            void *stream,
            TensorBase *tensor)
        {
            if (!tensor || !tensor->ensureOnHost(stream) ||
                !backend->synchronizeStream(stream, device.ordinal))
            {
                throw std::runtime_error(
                    "Could not stage arithmetic evidence to host");
            }
        }

        /** @brief Require exact equality for an in-backend arithmetic oracle. */
        void requireExactTensor(
            const TensorBase &expected,
            const TensorBase &actual,
            std::string_view context)
        {
            if (expected.numel() != actual.numel() ||
                std::memcmp(
                    expected.data(), actual.data(),
                    expected.numel() * sizeof(float)) != 0)
            {
                throw std::runtime_error(
                    std::string(context) + " is not byte-exact");
            }
        }

        /**
         * @brief Emit compact live regions for one prepared matrix.
         * @param writer Evidence owner receiving exact device bytes.
         * @param matrix Production-prepared matrix to observe.
         * @param source_format Canonical source-format storage contract.
         * @param variant Weight variant represented by @p matrix.
         * @param projection Stable gate, up, or down projection label.
         */
        void emitPreparedMatrixBlobs(
            EvidenceWriter &writer,
            const PreparedMatrix &matrix,
            const NativeVnniFormatInfo &source_format,
            int variant,
            std::string_view projection)
        {
            const auto sizes = nativeVnniPackedRegionSizes(
                static_cast<size_t>(matrix.descriptor.n),
                static_cast<size_t>(matrix.descriptor.k), source_format);
            const std::string prefix =
                "prepared.variant" + std::to_string(variant) + "." +
                std::string(projection);
            writer.emitDeviceBlob(
                prefix + ".payload.bin", matrix.descriptor.payload,
                sizes.payload_bytes, "bytes");
            writer.emitDeviceBlob(
                prefix + ".scales.bin", matrix.descriptor.scales,
                sizes.scales_bytes, "fp16_bits");
            if (sizes.mins_bytes != 0u)
            {
                writer.emitDeviceBlob(
                    prefix + ".mins.bin", matrix.descriptor.mins,
                    sizes.mins_bytes, "fp16_bits");
            }
            if (sizes.emins_bytes != 0u)
            {
                writer.emitDeviceBlob(
                    prefix + ".emins.bin", matrix.descriptor.emins,
                    sizes.emins_bytes, "u32");
            }
        }

        /** @brief Fill one deterministic hidden-state matrix on the host. */
        std::unique_ptr<TensorBase> makeHidden(
            int rows,
            size_t format_index,
            int first_logical_row = 0)
        {
            auto hidden = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), static_cast<size_t>(kDModel)});
            for (size_t index = 0; index < hidden->numel(); ++index)
            {
                const size_t source_index =
                    static_cast<size_t>(first_logical_row) * kDModel + index;
                hidden->mutable_data()[index] =
                    0.017f * std::sin(
                                 0.0041f * static_cast<float>(
                                               source_index + 11u +
                                               format_index)) -
                    0.009f * std::cos(
                                 0.0067f * static_cast<float>(
                                               source_index + 23u)) +
                    0.0007f * static_cast<float>(
                                  static_cast<int>(source_index % 31u) - 15);
            }
            return hidden;
        }

        /** @brief Create top-8 routes whose original slot order changes by row. */
        std::pair<std::unique_ptr<TensorBase>, std::unique_ptr<TensorBase>>
        makeRoutes(int rows, int first_logical_row = 0)
        {
            auto indices = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), static_cast<size_t>(kTopK)});
            auto weights = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), static_cast<size_t>(kTopK)});
            for (int row = 0; row < rows; ++row)
            {
                const int logical_row = first_logical_row + row;
                float weight_sum = 0.0f;
                for (int route = 0; route < kTopK; ++route)
                {
                    const size_t slot =
                        static_cast<size_t>(row * kTopK + route);
                    indices->mutable_data()[slot] = static_cast<float>(
                        (route * 5 + logical_row * 3 + 3) % kNumExperts);
                    const float raw_weight =
                        0.071f + 0.013f * static_cast<float>(
                                              (logical_row * 7 + route * 3 +
                                               2) %
                                              11);
                    weights->mutable_data()[slot] = raw_weight;
                    weight_sum += raw_weight;
                }
                for (int route = 0; route < kTopK; ++route)
                {
                    const size_t slot =
                        static_cast<size_t>(row * kTopK + route);
                    weights->mutable_data()[slot] /= weight_sum;
                }
            }
            return {std::move(indices), std::move(weights)};
        }

        /** @brief Execute the fused/split top-8 M=1 decode contract. */
        void runDecodeCase(
            IMoEKernel &kernel,
            IBackend *backend,
            DeviceId device,
            void *stream,
            DeviceWorkspaceManager &workspace,
            int gateup_table,
            int down_table,
            size_t format_index,
            int logical_row,
            EvidenceWriter &writer)
        {
            auto hidden = makeHidden(1, format_index, logical_row);
            auto [routing_indices, routing_weights] =
                makeRoutes(1, logical_row);

            std::array<std::unique_ptr<TensorBase>, kTopK> gate_owned;
            std::array<std::unique_ptr<TensorBase>, kTopK> up_owned;
            std::array<ITensor *, kTopK> gate_outputs{};
            std::array<ITensor *, kTopK> up_outputs{};
            for (int route = 0; route < kTopK; ++route)
            {
                gate_owned[static_cast<size_t>(route)] =
                    TestTensorFactory::createFP32(
                        {1u, static_cast<size_t>(kIntermediate)});
                up_owned[static_cast<size_t>(route)] =
                    TestTensorFactory::createFP32(
                        {1u, static_cast<size_t>(kIntermediate)});
                gate_outputs[static_cast<size_t>(route)] =
                    gate_owned[static_cast<size_t>(route)].get();
                up_outputs[static_cast<size_t>(route)] =
                    up_owned[static_cast<size_t>(route)].get();
            }
            auto split_output = TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(kDModel)});
            auto fused_output = TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(kDModel)});
            auto canonical = TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(kTopK),
                 static_cast<size_t>(kDModel)});
            std::fill_n(
                canonical->mutable_data(), canonical->numel(),
                std::numeric_limits<float>::quiet_NaN());

            std::vector<TensorBase *> device_inputs = {
                hidden.get(), routing_indices.get(), routing_weights.get(),
                split_output.get(), canonical.get()};
            for (auto &tensor : gate_owned)
                device_inputs.push_back(tensor.get());
            for (auto &tensor : up_owned)
                device_inputs.push_back(tensor.get());
            for (TensorBase *tensor : device_inputs)
            {
                if (!tensor->ensureOnDevice(device, stream))
                    throw std::runtime_error("Could not publish decode tensor");
            }

            if (!kernel.groupedExpertGateUpDecodeFromRouting(
                    hidden.get(), routing_indices.get(), gateup_table, kTopK,
                    gate_outputs.data(), up_outputs.data(), kDModel,
                    kIntermediate) ||
                !kernel.groupedExpertDownDecodeFromRouting(
                    gate_outputs.data(), up_outputs.data(),
                    routing_indices.get(), routing_weights.get(), down_table,
                    kTopK, split_output.get(), kDModel, kIntermediate) ||
                !kernel.groupedExpertDecodeFromRouting(
                    hidden.get(), routing_indices.get(), routing_weights.get(),
                    gateup_table, down_table, kTopK, fused_output.get(),
                    kDModel, kIntermediate, nullptr, canonical.get()))
            {
                throw std::runtime_error("Grouped decode evidence launch failed");
            }
            if (fused_output->gpu_data_ptr() != nullptr ||
                !fused_output->ensureOnDevice(device, stream) ||
                !kernel.reduceCanonicalRouteContributions(
                    canonical.get(), fused_output.get(), 1, kTopK, kDModel))
            {
                throw std::runtime_error(
                    "Canonical decode reduction contract failed");
            }

            const std::string case_label =
                logical_row == 0
                    ? "decode_m1"
                    : "decode_m1_row" + std::to_string(logical_row);
            writer.emitDeviceBlob(
                case_label + ".canonical.bin", canonical->gpu_data_ptr(),
                canonical->numel() * sizeof(float), "fp32", 1, kTopK,
                kDModel);
            writer.emitDeviceBlob(
                case_label + ".hidden_q8.bin",
                workspace.getBuffer(MoEWorkspaceBuffers::DECODE_HIDDEN_INT8),
                static_cast<size_t>(kDModel), "i8", 1, 1, kDModel);
            writer.emitDeviceBlob(
                case_label + ".hidden_q8_scales.bin",
                workspace.getBuffer(
                    MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES),
                static_cast<size_t>(kDModel / 32) * sizeof(float), "fp32",
                1, 1, kDModel / 32);
            writer.emitDeviceBlob(
                case_label + ".swiglu_q8.bin",
                workspace.getBuffer(MoEWorkspaceBuffers::DECODE_SWIGLU_INT8),
                static_cast<size_t>(kTopK * kIntermediate), "i8", 1,
                kTopK, kIntermediate);
            writer.emitDeviceBlob(
                case_label + ".swiglu_q8_scales.bin",
                workspace.getBuffer(
                    MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES),
                static_cast<size_t>(kTopK * (kIntermediate / 32)) *
                    sizeof(float),
                "fp32", 1, kTopK, kIntermediate / 32);
            observeTensor(backend, device, stream, split_output.get());
            observeTensor(backend, device, stream, fused_output.get());
            observeTensor(backend, device, stream, canonical.get());
            requireExactTensor(
                *split_output, *fused_output,
                "split and fused top-8 decode");

            for (int route = 0; route < kTopK; ++route)
            {
                observeTensor(
                    backend, device, stream,
                    gate_owned[static_cast<size_t>(route)].get());
                observeTensor(
                    backend, device, stream,
                    up_owned[static_cast<size_t>(route)].get());
                writer.emitTensor(
                    case_label, "gate.route" + std::to_string(route),
                    *gate_owned[static_cast<size_t>(route)]);
                writer.emitTensor(
                    case_label, "up.route" + std::to_string(route),
                    *up_owned[static_cast<size_t>(route)]);
            }
            writer.emitTensor(case_label, "split_output", *split_output);
            writer.emitTensor(case_label, "fused_output", *fused_output);
        }

        /** @brief Execute one masked canonical grouped-prefill transaction. */
        void runGroupedCase(
            IMoEKernel &kernel,
            IBackend *backend,
            DeviceId device,
            void *stream,
            DeviceWorkspaceManager &workspace,
            int gateup_table,
            int down_table,
            int rows,
            size_t format_index,
            EvidenceWriter &writer)
        {
            auto hidden = makeHidden(rows, format_index);
            auto [routing_indices, routing_weights] = makeRoutes(rows);
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), static_cast<size_t>(kDModel)});
            auto canonical = TestTensorFactory::createFP32(
                {static_cast<size_t>(rows), static_cast<size_t>(kTopK),
                 static_cast<size_t>(kDModel)});
            std::fill_n(
                canonical->mutable_data(), canonical->numel(),
                std::numeric_limits<float>::quiet_NaN());
            const std::array<TensorBase *, 4> device_inputs = {
                hidden.get(), routing_indices.get(), routing_weights.get(),
                canonical.get()};
            for (TensorBase *tensor : device_inputs)
            {
                if (!tensor->ensureOnDevice(device, stream))
                    throw std::runtime_error("Could not publish prefill tensor");
            }

            const std::array<uint8_t, kNumExperts> all_local = {
                1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u};
            if (!kernel.updateGroupedPrefillExpertMask(
                    all_local.data(), kNumExperts) ||
                !kernel.prepareExpertGroupsAsyncUsingPublishedMask(
                    routing_indices.get(), routing_weights.get(), rows,
                    kNumExperts, kTopK) ||
                !kernel.executeGroupedPrefillPipeline(
                    hidden.get(), output.get(), gateup_table, down_table, rows,
                    kDModel, kIntermediate, kNumExperts, kTopK,
                    canonical.get()))
            {
                throw std::runtime_error(
                    "Canonical grouped-prefill evidence launch failed");
            }
            if (output->gpu_data_ptr() != nullptr ||
                !output->ensureOnDevice(device, stream) ||
                !kernel.reduceCanonicalRouteContributions(
                    canonical.get(), output.get(), rows, kTopK, kDModel))
            {
                throw std::runtime_error(
                    "Canonical grouped-prefill reduction failed");
            }

            const std::string case_label =
                "prefill_m" + std::to_string(rows);
            const size_t route_slots =
                static_cast<size_t>(rows * kTopK);
            const int scale_width = kIntermediate / 32;
            std::vector<int> original_to_grouped(route_slots);
            std::vector<int8_t> grouped_q8(
                route_slots * static_cast<size_t>(kIntermediate));
            std::vector<float> grouped_scales(
                route_slots * static_cast<size_t>(scale_width));
            if (!backend->deviceToHostFast(
                    original_to_grouped.data(),
                    workspace.getBuffer(
                        MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED),
                    original_to_grouped.size() * sizeof(int),
                    device.ordinal, stream) ||
                !backend->deviceToHostFast(
                    grouped_q8.data(),
                    workspace.getBuffer(
                        MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8),
                    grouped_q8.size() * sizeof(int8_t),
                    device.ordinal, stream) ||
                !backend->deviceToHostFast(
                    grouped_scales.data(),
                    workspace.getBuffer(
                        MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES),
                    grouped_scales.size() * sizeof(float),
                    device.ordinal, stream) ||
                !backend->synchronizeStream(stream, device.ordinal))
            {
                throw std::runtime_error(
                    "Could not observe grouped SwiGLU publication");
            }

            // Physical expert grouping is backend-specific. Compare the live
            // values only after applying the production inverse permutation
            // that restores original router-slot order.
            std::vector<int8_t> canonical_q8(grouped_q8.size());
            std::vector<float> canonical_scales(grouped_scales.size());
            for (size_t original_slot = 0; original_slot < route_slots;
                 ++original_slot)
            {
                const int grouped_slot =
                    original_to_grouped[original_slot];
                if (grouped_slot < 0 ||
                    grouped_slot >= static_cast<int>(route_slots))
                {
                    throw std::runtime_error(
                        "Invalid original-to-grouped SwiGLU permutation");
                }
                std::memcpy(
                    canonical_q8.data() +
                        original_slot * static_cast<size_t>(kIntermediate),
                    grouped_q8.data() +
                        static_cast<size_t>(grouped_slot) * kIntermediate,
                    static_cast<size_t>(kIntermediate));
                std::memcpy(
                    canonical_scales.data() +
                        original_slot * static_cast<size_t>(scale_width),
                    grouped_scales.data() +
                        static_cast<size_t>(grouped_slot) * scale_width,
                    static_cast<size_t>(scale_width) * sizeof(float));
            }
            writer.emitHostBlob(
                case_label + ".swiglu_q8.bin", canonical_q8.data(),
                canonical_q8.size() * sizeof(int8_t), "i8", rows, kTopK,
                kIntermediate);
            writer.emitHostBlob(
                case_label + ".swiglu_q8_scales.bin",
                canonical_scales.data(),
                canonical_scales.size() * sizeof(float), "fp32", rows,
                kTopK, scale_width);
            writer.emitDeviceBlob(
                case_label + ".canonical.bin", canonical->gpu_data_ptr(),
                canonical->numel() * sizeof(float), "fp32", rows, kTopK,
                kDModel);

            observeTensor(backend, device, stream, output.get());
            observeTensor(backend, device, stream, canonical.get());
            writer.emitTensor(case_label, "folded_output", *output);
        }

        /** @brief Execute one format's complete production arithmetic witness. */
        void runEmitter(
            DeviceId device,
            std::string_view backend_label,
            std::string_view format_label,
            const std::filesystem::path &output_path)
        {
            IBackend *backend = getBackendFor(device);
            if (!backend || backend->deviceCount() <= device.ordinal)
            {
                throw std::runtime_error(
                    "Requested arithmetic-evidence device is unavailable");
            }
            const auto &format = quantizedMoEVerifierFormat(format_label);
            const size_t format_index = formatOrdinal(format_label);
            const NativeVnniFormatInfo *source_format =
                native_vnni_formats::forSourceIdentity(
                    format.source_codebook_id, format.source_is_superblock);
            if (!source_format)
            {
                throw std::runtime_error(
                    "Canonical format has no source metadata: " +
                    std::string(format_label));
            }

            ExplicitStream stream_owner(backend, device);
            void *const stream = stream_owner.get();
            auto kernel_owner =
                llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            if (!kernel_owner)
                throw std::runtime_error("Could not create production MoE kernel");
            kernel_owner->setGPUStream(stream);

            constexpr int max_rows = kGroupedRows.back();
            const auto requirements = device.is_cuda()
                                          ? MoEWorkspaceBuffers::cudaMoE(
                                                max_rows, kDModel,
                                                kIntermediate, kNumExperts,
                                                kTopK)
                                          : MoEWorkspaceBuffers::rocmMoE(
                                                max_rows, kDModel,
                                                kIntermediate, kNumExperts,
                                                kTopK);
            DeviceWorkspaceManager workspace(
                device, requirements.total_bytes_with_alignment() +
                            4u * 1024u * 1024u);
            if (!workspace.allocate(requirements))
                throw std::runtime_error("Could not allocate MoE workspace");
            auto *workspace_consumer =
                dynamic_cast<IWorkspaceConsumer *>(kernel_owner.get());
            if (!workspace_consumer)
                throw std::runtime_error("MoE kernel is not workspace-aware");
            workspace_consumer->bindWorkspace(&workspace);

            EvidenceWriter writer(
                backend, device, stream, std::string(backend_label),
                std::string(format_label), output_path);
            std::array<PreparedExpert, kWeightVariants> variants;
            const uint64_t backend_offset = device.is_cuda() ? 0u : 100000u;
            for (int variant = 0; variant < kWeightVariants; ++variant)
            {
                const uint32_t seed_base =
                    910000u + static_cast<uint32_t>(variant * 100u);
                const uint64_t id_base =
                    2200000u + backend_offset +
                    static_cast<uint64_t>(
                        format_index * 100u + variant * 3u);
                const std::string name_base =
                    std::string(backend_label) + ".cross_vendor." +
                    std::string(format_label) + ".variant" +
                    std::to_string(variant);
                variants[static_cast<size_t>(variant)].gate = prepareMatrix(
                    format, device, stream, kIntermediate, kDModel,
                    seed_base + 1u, name_base + ".gate",
                    ModelContextId{id_base + 1u});
                variants[static_cast<size_t>(variant)].up = prepareMatrix(
                    format, device, stream, kIntermediate, kDModel,
                    seed_base + 2u, name_base + ".up",
                    ModelContextId{id_base + 2u});
                variants[static_cast<size_t>(variant)].down = prepareMatrix(
                    format, device, stream, kDModel, kIntermediate,
                    seed_base + 3u, name_base + ".down",
                    ModelContextId{id_base + 3u});

                writer.emitDescriptor(
                    variant, "gate",
                    variants[static_cast<size_t>(variant)].gate.descriptor,
                    *source_format);
                writer.emitDescriptor(
                    variant, "up",
                    variants[static_cast<size_t>(variant)].up.descriptor,
                    *source_format);
                writer.emitDescriptor(
                    variant, "down",
                    variants[static_cast<size_t>(variant)].down.descriptor,
                    *source_format);
            }

            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> gate_descs{};
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> up_descs{};
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> down_descs{};
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                const PreparedExpert &variant = variants[static_cast<size_t>(
                    expert % kWeightVariants)];
                gate_descs[static_cast<size_t>(expert)] =
                    variant.gate.descriptor;
                up_descs[static_cast<size_t>(expert)] = variant.up.descriptor;
                down_descs[static_cast<size_t>(expert)] =
                    variant.down.descriptor;
            }

            IMoEKernel &kernel = *kernel_owner;
            const int gateup_table =
                kernel.uploadGroupedExpertGateUpDescriptorTables(
                    gate_descs.data(), up_descs.data(), kNumExperts,
                    kDModel, kIntermediate);
            const int down_table =
                kernel.uploadGroupedExpertDownDescriptorTable(
                    down_descs.data(), kNumExperts, kDModel, kIntermediate);
            if (gateup_table < 0 || down_table < 0)
                throw std::runtime_error(
                    "Could not upload grouped descriptor tables");

            // Both variants participate in grouped rows. Retain both prepared
            // byte contracts so a placement-dependent failure is attributable
            // before it propagates through the fused expert arithmetic.
            for (int variant = 0; variant < kWeightVariants; ++variant)
            {
                const PreparedExpert &prepared =
                    variants[static_cast<size_t>(variant)];
                emitPreparedMatrixBlobs(
                    writer, prepared.gate, *source_format, variant, "gate");
                emitPreparedMatrixBlobs(
                    writer, prepared.up, *source_format, variant, "up");
                emitPreparedMatrixBlobs(
                    writer, prepared.down, *source_format, variant, "down");
            }
            runDecodeCase(
                kernel, backend, device, stream, workspace, gateup_table,
                down_table, format_index, 0, writer);
            runDecodeCase(
                kernel, backend, device, stream, workspace, gateup_table,
                down_table, format_index, 1, writer);
            for (const int rows : kGroupedRows)
            {
                runGroupedCase(
                    kernel, backend, device, stream, workspace, gateup_table,
                    down_table, rows, format_index, writer);
            }
            writer.finalize();
        }

        /** @brief Print the canonical MoE source-format inventory. */
        void listFormats()
        {
            for (const auto &format : quantizedMoEVerifierFormats())
                std::cout << format.label << '\n';
        }
    } // namespace
} // namespace llaminar2::test

/**
 * @brief Emit one backend's process-isolated arithmetic evidence.
 * @param argc Accepts `--list-formats` or `FORMAT OUTPUT.csv`.
 * @param argv Command-line arguments described by @p argc.
 * @return Zero on complete evidence, nonzero on invalid input or execution.
 */
int main(int argc, char **argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--list-formats")
    {
        llaminar2::test::listFormats();
        return 0;
    }
    if (argc != 3)
    {
        std::cerr << "usage: " << argv[0]
                  << " --list-formats | FORMAT OUTPUT.csv\n";
        return 2;
    }

    try
    {
#if defined(LLAMINAR_CROSS_VENDOR_ARITHMETIC_CUDA)
        llaminar2::test::runEmitter(
            llaminar2::DeviceId::cuda(0), "cuda", argv[1], argv[2]);
#elif defined(LLAMINAR_CROSS_VENDOR_ARITHMETIC_ROCM)
        llaminar2::test::runEmitter(
            llaminar2::DeviceId::rocm(0), "rocm", argv[1], argv[2]);
#else
#error "Cross-vendor arithmetic emitter requires one backend definition"
#endif
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "cross-vendor arithmetic evidence failed: "
                  << error.what() << '\n';
        return 1;
    }
}
