/**
 * @file Test__ExpertTierGpuBlobTransferIntegration.cpp
 * @brief Real-device CUDA/ROCm packed and floating expert relay proof.
 *
 * The test transfers a production-shaped separated NativeVNNI projection in
 * CUDA-to-ROCm, ROCm-to-CUDA, and same-backend/no-P2P directions, then repeats
 * the cross-backend production lane for contiguous FP16, BF16, and FP32
 * payloads. It validates every byte, exercises the double-buffered async-DMA
 * epoch state machine, and proves independent work on both devices completes
 * while migration remains a background transaction.
 */

#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/moe/ExpertTierGpuBlobTransferLane.h"
#include "execution/moe/MoEOverlayGpuRemoteProjectionEndpoint.h"
#include "transfer/TransferEngine.h"
#include "kernels/cuda/gemm/CUDAFloatingPointGemmKernel.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmFloatingPointGemmKernel.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>

#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Enable PerfStats for the scoped movement certificate. */
        class ScopedHeterogeneousBlobPerfStats final
        {
        public:
            /** @brief Save process configuration, enable counters, and clear rows. */
            ScopedHeterogeneousBlobPerfStats()
            {
                if (const char *value =
                        std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
                {
                    had_value_ = true;
                    value_ = value;
                }
                setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
                mutableDebugEnv().reload();
                PerfStatsCollector::reset();
            }

            /** @brief Restore process configuration and discard test evidence. */
            ~ScopedHeterogeneousBlobPerfStats()
            {
                if (had_value_)
                    setenv(
                        "LLAMINAR_PERF_STATS_SUMMARY",
                        value_.c_str(),
                        1);
                else
                    unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
                mutableDebugEnv().reload();
                PerfStatsCollector::reset();
            }

            ScopedHeterogeneousBlobPerfStats(
                const ScopedHeterogeneousBlobPerfStats &) = delete;
            ScopedHeterogeneousBlobPerfStats &operator=(
                const ScopedHeterogeneousBlobPerfStats &) = delete;

        private:
            bool had_value_ = false;
            std::string value_;
        };

        /**
         * @brief Own the four separated arrays for one GPU descriptor.
         *
         * Allocations and releases are routed through the exact backend so this
         * ordinary C++ test never includes CUDA and HIP runtime headers in the
         * same translation unit.
         */
        class DevicePackedProjection final
        {
        public:
            /**
             * @brief Allocate a Q4-style asymmetric projection on one endpoint.
             * @param backend Backend owning all allocations.
             * @param device Exact device identity.
             * @param n Output rows.
             * @param k Input columns, divisible by 32.
             */
            DevicePackedProjection(
                IBackend &backend,
                DeviceId device,
                int n,
                int k)
                : backend_(backend),
                  device_(device),
                  n_(n),
                  k_(k),
                  block_count_(
                      static_cast<std::size_t>(n) *
                      static_cast<std::size_t>(k / 32)),
                  payload_bytes_(block_count_ * 16),
                  metadata_bytes_(block_count_ * sizeof(std::uint16_t))
            {
                payload_ = static_cast<std::uint8_t *>(
                    backend_.allocate(payload_bytes_, device_.gpu_ordinal()));
                scales_ = static_cast<std::uint16_t *>(
                    backend_.allocate(metadata_bytes_, device_.gpu_ordinal()));
                mins_ = static_cast<std::uint16_t *>(
                    backend_.allocate(metadata_bytes_, device_.gpu_ordinal()));
                if (!payload_ || !scales_ || !mins_)
                    throw std::runtime_error(
                        "Could not allocate heterogeneous packed projection");
            }

            /** @brief Release all device regions through their owning backend. */
            ~DevicePackedProjection()
            {
                const int ordinal = device_.gpu_ordinal();
                if (payload_)
                    backend_.free(payload_, ordinal);
                if (scales_)
                    backend_.free(scales_, ordinal);
                if (mins_)
                    backend_.free(mins_, ordinal);
            }

            DevicePackedProjection(const DevicePackedProjection &) = delete;
            DevicePackedProjection &operator=(
                const DevicePackedProjection &) = delete;

            /** @brief Return the byte-exact common descriptor used by the lane. */
            [[nodiscard]] GpuExpertPackedDescriptor descriptor() const noexcept
            {
                return {
                    .ptrs = {
                        .d_vnni = payload_,
                        .d_scales = scales_,
                        .d_mins = mins_,
                    },
                    .n = n_,
                    .k = k_,
                    .blocks_per_row = static_cast<std::uint32_t>(k_ / 32),
                    .codebook_id = 5,
                    .payload_bytes_per_block = 16,
                    .is_asymmetric = true,
                    .has_emins = false,
                    .vnni_bytes = payload_bytes_,
                    .scales_bytes = metadata_bytes_,
                    .mins_bytes = metadata_bytes_,
                    .emins_bytes = 0,
                };
            }

            /**
             * @brief Enqueue all host source bytes on one exact producer stream.
             * @param payload Host payload bytes.
             * @param scales Host scale words.
             * @param mins Host minimum words.
             * @param stream Explicit stream owned by this device.
             * @return Whether every H2D copy was accepted.
             */
            bool upload(
                const std::vector<std::uint8_t> &payload,
                const std::vector<std::uint16_t> &scales,
                const std::vector<std::uint16_t> &mins,
                void *stream) noexcept
            {
                if (payload.size() != payload_bytes_ ||
                    scales.size() * sizeof(std::uint16_t) != metadata_bytes_ ||
                    mins.size() * sizeof(std::uint16_t) != metadata_bytes_)
                {
                    return false;
                }
                const int ordinal = device_.gpu_ordinal();
                return backend_.hostToDeviceOnStream(
                           payload_, payload.data(), payload_bytes_, ordinal, stream) &&
                       backend_.hostToDeviceOnStream(
                           scales_, scales.data(), metadata_bytes_, ordinal, stream) &&
                       backend_.hostToDeviceOnStream(
                           mins_, mins.data(), metadata_bytes_, ordinal, stream);
            }

            /**
             * @brief Observe final destination bytes after the lane reports ready.
             * @param stream Explicit diagnostic stream.
             * @param payload Output payload storage.
             * @param scales Output scale storage.
             * @param mins Output minimum storage.
             * @return Whether all diagnostic D2H operations completed.
             */
            bool download(
                void *stream,
                std::vector<std::uint8_t> &payload,
                std::vector<std::uint16_t> &scales,
                std::vector<std::uint16_t> &mins) noexcept
            {
                payload.resize(payload_bytes_);
                scales.resize(block_count_);
                mins.resize(block_count_);
                const int ordinal = device_.gpu_ordinal();
                return backend_.deviceToHost(
                           payload.data(), payload_, payload_bytes_, ordinal, stream) &&
                       backend_.deviceToHost(
                           scales.data(), scales_, metadata_bytes_, ordinal, stream) &&
                       backend_.deviceToHost(
                           mins.data(), mins_, metadata_bytes_, ordinal, stream);
            }

            /** @brief Return exact total bytes in all non-empty regions. */
            [[nodiscard]] std::size_t totalBytes() const noexcept
            {
                return payload_bytes_ + 2 * metadata_bytes_;
            }

        private:
            IBackend &backend_;
            DeviceId device_;
            int n_ = 0;
            int k_ = 0;
            std::size_t block_count_ = 0;
            std::size_t payload_bytes_ = 0;
            std::size_t metadata_bytes_ = 0;
            std::uint8_t *payload_ = nullptr;
            std::uint16_t *scales_ = nullptr;
            std::uint16_t *mins_ = nullptr;
        };

        /** @brief Own one opaque contiguous floating projection on a GPU. */
        class DeviceContiguousProjection final
        {
        public:
            /**
             * @brief Allocate the complete representation on one exact device.
             * @param backend Runtime authority for allocation and copies.
             * @param device Exact CUDA or ROCm endpoint.
             * @param bytes Non-zero size of the FP16/BF16/FP32 byte payload.
             */
            DeviceContiguousProjection(
                IBackend &backend,
                DeviceId device,
                std::size_t bytes)
                : backend_(backend), device_(device), bytes_(bytes)
            {
                if (bytes_ == 0)
                    throw std::invalid_argument(
                        "Contiguous floating projection cannot be empty");
                data_ = static_cast<std::uint8_t *>(
                    backend_.allocate(bytes_, device_.gpu_ordinal()));
                if (!data_)
                    throw std::runtime_error(
                        "Could not allocate heterogeneous floating projection");
            }

            /** @brief Release the representation through its owning runtime. */
            ~DeviceContiguousProjection()
            {
                if (data_)
                    backend_.free(data_, device_.gpu_ordinal());
            }

            DeviceContiguousProjection(
                const DeviceContiguousProjection &) = delete;
            DeviceContiguousProjection &operator=(
                const DeviceContiguousProjection &) = delete;

            /**
             * @brief Publish source bytes on the exact producer stream.
             * @param bytes Host representation to upload byte-for-byte.
             * @param stream Non-null stream owned by this endpoint.
             * @return Whether the asynchronous upload was accepted.
             */
            bool upload(
                const std::vector<std::uint8_t> &bytes,
                void *stream) noexcept
            {
                return bytes.size() == bytes_ && stream &&
                       backend_.hostToDeviceOnStream(
                           data_,
                           bytes.data(),
                           bytes_,
                           device_.gpu_ordinal(),
                           stream);
            }

            /**
             * @brief Observe destination bytes after the lane reports ready.
             * @param stream Exact diagnostic stream on this endpoint.
             * @param bytes Destination host buffer, resized by this method.
             * @return Whether the diagnostic D2H copy completed.
             */
            bool download(
                void *stream,
                std::vector<std::uint8_t> &bytes) noexcept
            {
                if (!stream)
                    return false;
                bytes.resize(bytes_);
                return backend_.deviceToHost(
                    bytes.data(),
                    data_,
                    bytes_,
                    device_.gpu_ordinal(),
                    stream);
            }

            /** @return Mutable storage used only as a lane destination. */
            [[nodiscard]] void *data() noexcept { return data_; }

            /** @return Immutable storage used only as a lane source. */
            [[nodiscard]] const void *data() const noexcept { return data_; }

            /** @return Complete contiguous representation size. */
            [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

        private:
            IBackend &backend_;
            DeviceId device_;
            std::size_t bytes_ = 0;
            std::uint8_t *data_ = nullptr;
        };

        /** @brief Own a small allocation used as independent device work. */
        class DeviceCopyWitness final
        {
        public:
            /** @brief Allocate source and destination regions on one endpoint. */
            DeviceCopyWitness(IBackend &backend, DeviceId device)
                : backend_(backend), device_(device)
            {
                constexpr std::size_t bytes = 4 * 1024 * 1024;
                source_ = backend_.allocate(bytes, device_.gpu_ordinal());
                destination_ = backend_.allocate(bytes, device_.gpu_ordinal());
                if (!source_ || !destination_)
                    throw std::runtime_error("Could not allocate independent work witness");
            }

            /** @brief Release witness allocations after its event is complete. */
            ~DeviceCopyWitness()
            {
                const int ordinal = device_.gpu_ordinal();
                if (source_)
                    backend_.free(source_, ordinal);
                if (destination_)
                    backend_.free(destination_, ordinal);
            }

            DeviceCopyWitness(const DeviceCopyWitness &) = delete;
            DeviceCopyWitness &operator=(const DeviceCopyWitness &) = delete;

            /** @brief Enqueue the independent copy on the provided stream. */
            bool enqueue(void *stream) noexcept
            {
                return backend_.deviceCopyAsync(
                    destination_,
                    source_,
                    4 * 1024 * 1024,
                    device_.gpu_ordinal(),
                    stream);
            }

        private:
            IBackend &backend_;
            DeviceId device_;
            void *source_ = nullptr;
            void *destination_ = nullptr;
        };

        /** @brief Shared source/destination epochs used by one relay fixture. */
        struct RetainedRelayEpochPair
        {
            std::shared_ptr<MappedTransferProgressEpoch> source;
            std::shared_ptr<MappedTransferProgressEpoch> destination;
        };

        /** @brief Materialize the exact four permanent slots owned by one lane. */
        RetainedRelayEpochPair makeRetainedRelayEpochs(
            DeviceId source_device,
            DeviceId destination_device,
            std::size_t staging_bytes,
            const std::string &identity)
        {
            return {
                .source = MappedTransferProgressEpoch::create({
                    .device = source_device,
                    .slot_capacity = 2u,
                    .execution_lane_capacity = 2u,
                    .execution_streams =
                        TransferEngine::instance()
                            .allocatePersistentTransferExecutionLanes(
                                2u,
                                source_device,
                                "blob_progress_source:" + identity),
                    .maximum_bytes = staging_bytes,
                    .name = "blob_source:" + identity,
                    .perf_device = identity,
                }),
                .destination = MappedTransferProgressEpoch::create({
                    .device = destination_device,
                    .slot_capacity = 2u,
                    .execution_lane_capacity = 2u,
                    .execution_streams =
                        TransferEngine::instance()
                            .allocatePersistentTransferExecutionLanes(
                                2u,
                                destination_device,
                                "blob_progress_destination:" + identity),
                    .maximum_bytes = staging_bytes,
                    .name = "blob_destination:" + identity,
                    .perf_device = identity,
                }),
            };
        }

        /**
         * @brief Independent captured work plus an explicit maintenance pump.
         *
         * The one-node graph models an operation with its own concurrent stream
         * pool. Progress is submitted separately after the graph launch, so the
         * fixture proves neither side captures or waits for the other's events.
         */
        class CapturedProgressReplay final
        {
        public:
            /** @brief Capture exactly one inference-like primary node. */
            CapturedProgressReplay(
                IWorkerGPUContext &context,
                std::shared_ptr<MappedTransferProgressEpoch> epoch,
                std::string identity)
                : context_(context),
                  epoch_(std::move(epoch)),
                  device_(epoch_ ? epoch_->device() : DeviceId::invalid()),
                  backend_(getBackendFor(device_))
            {
                if (!epoch_ || !device_.is_gpu() || !backend_)
                    throw std::invalid_argument(
                        "Captured progress replay requires one live GPU epoch");

                context_.submitAndWait(
                    [this, identity = std::move(identity)]
                    {
                        primary_stream_ =
                            context_.getOrCreateAuxiliaryStream(
                                "captured_blob_progress:" + identity);
                        terminal_event_ = context_.createEvent();
                        witness_ = backend_->allocate(
                            kWitnessBytes, device_.gpu_ordinal());
                        graph_ = context_.createGraphCapture(primary_stream_);
                        if (!primary_stream_ || !terminal_event_ || !witness_ ||
                            !graph_ || !graph_->beginCapture())
                        {
                            throw std::runtime_error(
                                "Could not materialize captured blob progress replay");
                        }

                        bool capture_open = true;
                        try
                        {
                            if (!backend_->memset(
                                    witness_,
                                    0x5a,
                                    kWitnessBytes,
                                    device_.gpu_ordinal(),
                                    primary_stream_) ||
                                !graph_->endCapture())
                            {
                                throw std::runtime_error(
                                    "Could not record captured blob progress replay");
                            }
                            capture_open = false;
                            if (graph_->nodeCount() != 1u ||
                                !graph_->instantiate())
                            {
                                throw std::runtime_error(
                                    "Captured blob progress replay has no complete executable");
                            }
                        }
                        catch (...)
                        {
                            if (capture_open)
                                (void)graph_->endCapture();
                            throw;
                        }
                    });
            }

            /** @brief Drain the final replay, then retire graph-owned handles. */
            ~CapturedProgressReplay()
            {
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::seconds(10);
                if (!awaitIdle(deadline))
                    std::terminate();
                try
                {
                    context_.submitAndWait(
                        [this]
                        {
                            graph_.reset();
                            if (terminal_event_)
                            {
                                context_.destroyEvent(terminal_event_);
                                terminal_event_ = nullptr;
                            }
                            if (witness_)
                            {
                                backend_->free(
                                    witness_, device_.gpu_ordinal());
                                witness_ = nullptr;
                            }
                        });
                }
                catch (...)
                {
                    std::terminate();
                }
            }

            CapturedProgressReplay(const CapturedProgressReplay &) = delete;
            CapturedProgressReplay &operator=(
                const CapturedProgressReplay &) = delete;

            /** @brief Enqueue one complete native graph without waiting. */
            [[nodiscard]] bool launch() noexcept
            {
                bool launched = false;
                try
                {
                    context_.submitAndWait(
                        [this, &launched]
                        {
                            launched = graph_ && graph_->launch() &&
                                context_.recordEventChecked(
                                    terminal_event_, primary_stream_) &&
                                epoch_->submitOutstandingProgress();
                            if (launched)
                                launch_pending_ = true;
                        });
                }
                catch (...)
                {
                    return false;
                }
                return launched;
            }

            /** @brief Poll the exact terminal event without synchronizing. */
            [[nodiscard]] bool awaitIdle(
                std::chrono::steady_clock::time_point deadline) noexcept
            {
                if (!launch_pending_)
                    return true;
                bool ready = false;
                while (!ready && std::chrono::steady_clock::now() < deadline)
                {
                    if (!context_.queryEventChecked(terminal_event_, ready))
                        return false;
                    if (!ready)
                        std::this_thread::yield();
                }
                if (ready)
                    launch_pending_ = false;
                return ready;
            }

        private:
            static constexpr std::size_t kWitnessBytes = 4096u;
            IWorkerGPUContext &context_;
            std::shared_ptr<MappedTransferProgressEpoch> epoch_;
            DeviceId device_ = DeviceId::invalid();
            IBackend *backend_ = nullptr;
            std::unique_ptr<IGPUGraphCapture> graph_;
            void *primary_stream_ = nullptr;
            void *terminal_event_ = nullptr;
            void *witness_ = nullptr;
            bool launch_pending_ = false;
        };

        /** @brief Non-blockingly close an upload event before bank publication. */
        bool awaitSetupEvent(
            IWorkerGPUContext &context,
            void *event,
            std::chrono::steady_clock::time_point deadline)
        {
            bool ready = false;
            while (!ready && std::chrono::steady_clock::now() < deadline)
            {
                if (!context.queryEventChecked(event, ready))
                    return false;
                if (!ready)
                    std::this_thread::yield();
            }
            return ready;
        }

        /**
         * @brief Run one direction of the real-device GPU host relay.
         * @param source_device Exact CUDA or ROCm source.
         * @param destination_device Exact CUDA or ROCm destination.
         * @param seed Deterministic byte-pattern seed.
         * @param relay_kind Topology fact requiring the explicit host relay.
         */
        void runHeterogeneousBlobDirection(
            DeviceId source_device,
            DeviceId destination_device,
            std::uint32_t seed,
            ExpertTierGpuBlobRelayKind relay_kind =
                ExpertTierGpuBlobRelayKind::CrossBackend)
        {
            IBackend *source_backend = getBackendFor(source_device);
            IBackend *destination_backend = getBackendFor(destination_device);
            ASSERT_NE(source_backend, nullptr);
            ASSERT_NE(destination_backend, nullptr);

            constexpr int n = 1024;
            constexpr int k = 256;
            constexpr std::size_t block_count =
                static_cast<std::size_t>(n) * (k / 32);
            std::vector<std::uint8_t> expected_payload(block_count * 16);
            std::vector<std::uint16_t> expected_scales(block_count);
            std::vector<std::uint16_t> expected_mins(block_count);
            for (std::size_t index = 0; index < expected_payload.size(); ++index)
            {
                expected_payload[index] = static_cast<std::uint8_t>(
                    (index * 37u + seed * 11u + 3u) & 0xffu);
            }
            for (std::size_t index = 0; index < block_count; ++index)
            {
                expected_scales[index] = static_cast<std::uint16_t>(
                    0x2400u + ((index * 13u + seed) & 0x07ffu));
                expected_mins[index] = static_cast<std::uint16_t>(
                    0xa000u + ((index * 17u + seed) & 0x07ffu));
            }

            DevicePackedProjection source(
                *source_backend,
                source_device,
                n,
                k);
            DevicePackedProjection destination(
                *destination_backend,
                destination_device,
                n,
                k);

            IWorkerGPUContext &source_context =
                GPUDeviceContextPool::instance().getContext(source_device);
            IWorkerGPUContext &destination_context =
                GPUDeviceContextPool::instance().getContext(destination_device);
            const std::string identity =
                source_device.to_string() + "_to_" +
                destination_device.to_string();
            void *source_producer_stream =
                source_context.getOrCreateAuxiliaryStream(
                    "heterogeneous_blob_test_producer:" + identity);
            void *source_inference_stream =
                source_context.getOrCreateAuxiliaryStream(
                    "heterogeneous_blob_test_inference:" + identity);
            void *destination_inference_stream =
                destination_context.getOrCreateAuxiliaryStream(
                    "heterogeneous_blob_test_inference:" + identity);
            ASSERT_NE(source_producer_stream, nullptr);
            ASSERT_NE(source_inference_stream, nullptr);
            ASSERT_NE(destination_inference_stream, nullptr);

            ASSERT_TRUE(source.upload(
                expected_payload,
                expected_scales,
                expected_mins,
                source_producer_stream));
            void *source_ready_event =
                source_backend->createEvent(source_device.gpu_ordinal());
            ASSERT_NE(source_ready_event, nullptr);
            ASSERT_TRUE(source_backend->recordEvent(
                source_ready_event,
                source_device.gpu_ordinal(),
                source_producer_stream));
            const auto setup_deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            ASSERT_TRUE(awaitSetupEvent(
                source_context, source_ready_event, setup_deadline));

            constexpr std::size_t staging_bytes = 4096u;
            auto progress_epochs = makeRetainedRelayEpochs(
                source_device,
                destination_device,
                staging_bytes,
                identity + ":" + std::to_string(seed));
            const auto source_mapped_staging =
                TransferEngine::instance().allocateMappedHostTransferSlices(
                    staging_bytes, 2u, source_device);
            const auto destination_mapped_staging =
                TransferEngine::instance().allocateMappedHostTransferSlices(
                    staging_bytes, 2u, destination_device);

            ExpertTierGpuBlobTransferLane lane({
                .source_device = source_device,
                .destination_device = destination_device,
                .relay_kind = relay_kind,
                .staging_capacity_bytes = staging_bytes,
                .source_mapped_staging = {
                    source_mapped_staging[0],
                    source_mapped_staging[1],
                },
                .destination_mapped_staging = {
                    destination_mapped_staging[0],
                    destination_mapped_staging[1],
                },
                .source_progress_epoch = progress_epochs.source,
                .destination_progress_epoch = progress_epochs.destination,
                .lane_name = "integration:" + identity,
                .perf_device = identity,
                .collect_timing_measurements = true,
            });
            std::string error;
            ASSERT_TRUE(lane.materialize(&error)) << error;
            ASSERT_TRUE(lane.start(
                source.descriptor(),
                destination.descriptor(),
                ExpertTierSourceReadiness::publishedResidencyBank(seed + 1u),
                &error))
                << error;
            ASSERT_EQ(
                lane.progress(),
                ExpertTierGpuBlobTransferProgress::Pending);
            CapturedProgressReplay source_progress_replay(
                source_context,
                progress_epochs.source,
                identity + ":source");
            CapturedProgressReplay destination_progress_replay(
                destination_context,
                progress_epochs.destination,
                identity + ":destination");

            DeviceCopyWitness source_witness(
                *source_backend,
                source_device);
            DeviceCopyWitness destination_witness(
                *destination_backend,
                destination_device);
            ASSERT_TRUE(source_witness.enqueue(source_inference_stream));
            ASSERT_TRUE(destination_witness.enqueue(destination_inference_stream));
            void *source_inference_event =
                source_backend->createEvent(source_device.gpu_ordinal());
            void *destination_inference_event =
                destination_backend->createEvent(
                    destination_device.gpu_ordinal());
            ASSERT_NE(source_inference_event, nullptr);
            ASSERT_NE(destination_inference_event, nullptr);
            ASSERT_TRUE(source_backend->recordEvent(
                source_inference_event,
                source_device.gpu_ordinal(),
                source_inference_stream));
            ASSERT_TRUE(destination_backend->recordEvent(
                destination_inference_event,
                destination_device.gpu_ordinal(),
                destination_inference_stream));

            bool source_inference_ready = false;
            bool destination_inference_ready = false;
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while ((!source_inference_ready || !destination_inference_ready) &&
                   std::chrono::steady_clock::now() < deadline)
            {
                ASSERT_TRUE(source_context.queryEventChecked(
                    source_inference_event,
                    source_inference_ready));
                ASSERT_TRUE(destination_context.queryEventChecked(
                    destination_inference_event,
                    destination_inference_ready));
                std::this_thread::yield();
            }
            EXPECT_TRUE(source_inference_ready);
            EXPECT_TRUE(destination_inference_ready);
            EXPECT_EQ(
                lane.progress(),
                ExpertTierGpuBlobTransferProgress::Pending)
                << "Independent device work must not wait for host-pumped migration";

            while (lane.progress() ==
                       ExpertTierGpuBlobTransferProgress::Pending &&
                   std::chrono::steady_clock::now() < deadline)
            {
                ASSERT_TRUE(source_progress_replay.launch());
                ASSERT_TRUE(destination_progress_replay.launch());
                ASSERT_NE(
                    lane.poll(&error),
                    ExpertTierGpuBlobTransferProgress::Failed)
                    << error;
                std::this_thread::yield();
            }
            ASSERT_EQ(
                lane.progress(),
                ExpertTierGpuBlobTransferProgress::Ready)
                << error;
            ASSERT_TRUE(source_progress_replay.awaitIdle(deadline));
            ASSERT_TRUE(destination_progress_replay.awaitIdle(deadline));

            const auto stats = lane.stats();
            EXPECT_EQ(stats.transfers_started, 1u);
            EXPECT_EQ(stats.transfers_completed, 1u);
            EXPECT_EQ(stats.bytes_submitted, source.totalBytes());
            EXPECT_EQ(stats.host_relay_bytes, source.totalBytes());
            EXPECT_EQ(stats.source_d2h_submissions, stats.chunks_submitted);
            EXPECT_EQ(
                stats.source_progress_kernel_submissions,
                stats.chunks_submitted);
            EXPECT_EQ(
                stats.destination_h2d_submissions,
                stats.chunks_submitted);
            EXPECT_EQ(
                stats.destination_progress_kernel_submissions,
                stats.chunks_submitted);
            EXPECT_EQ(stats.chunks_completed, stats.chunks_submitted);
            EXPECT_EQ(stats.maximum_in_flight_chunks, 2u);
            EXPECT_EQ(stats.failed_transfers, 0u);
            EXPECT_EQ(stats.timing_measurement_failures, 0u);
            EXPECT_TRUE(stats.last_measurement.valid());
            EXPECT_EQ(stats.last_measurement.bytes, source.totalBytes());
            EXPECT_EQ(stats.last_measurement.device_nanoseconds, 0u);
            EXPECT_GT(stats.last_measurement.host_nanoseconds, 0u);
            EXPECT_GT(
                stats.last_max_source_dma_residence_nanoseconds,
                0u);
            EXPECT_GT(
                stats.last_max_destination_dma_residence_nanoseconds,
                0u);
            EXPECT_GT(
                stats.last_max_maintenance_poll_gap_nanoseconds,
                0u);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);

            std::vector<std::uint8_t> actual_payload;
            std::vector<std::uint16_t> actual_scales;
            std::vector<std::uint16_t> actual_mins;
            ASSERT_TRUE(destination.download(
                destination_inference_stream,
                actual_payload,
                actual_scales,
                actual_mins));
            EXPECT_EQ(actual_payload, expected_payload);
            EXPECT_EQ(actual_scales, expected_scales);
            EXPECT_EQ(actual_mins, expected_mins);

            source_backend->destroyEvent(
                source_inference_event,
                source_device.gpu_ordinal());
            destination_backend->destroyEvent(
                destination_inference_event,
                destination_device.gpu_ordinal());
            source_backend->destroyEvent(
                source_ready_event,
                source_device.gpu_ordinal());
        }

        /**
         * @brief Run one cross-vendor contiguous floating projection movement.
         *
         * The relay sees only a byte span. A non-aligned element count forces a
         * partial final staging chunk, while independent copies on both devices
         * prove the background lane never inserts work into inference streams.
         *
         * @param source_device CUDA or ROCm source.
         * @param destination_device Opposite-backend destination.
         * @param precision_name Stable FP16/BF16/FP32 evidence identity.
         * @param element_bytes Bytes in one scalar representation.
         * @param seed Deterministic arbitrary-bit pattern seed.
         */
        void runHeterogeneousContiguousDirection(
            DeviceId source_device,
            DeviceId destination_device,
            const char *precision_name,
            std::size_t element_bytes,
            std::uint32_t seed)
        {
            ASSERT_TRUE(element_bytes == 2 || element_bytes == 4);
            IBackend *source_backend = getBackendFor(source_device);
            IBackend *destination_backend = getBackendFor(destination_device);
            ASSERT_NE(source_backend, nullptr);
            ASSERT_NE(destination_backend, nullptr);

            constexpr std::size_t element_count = 32771;
            constexpr std::size_t staging_bytes = 4096;
            const std::size_t projection_bytes =
                element_count * element_bytes;
            std::vector<std::uint8_t> expected(projection_bytes);
            for (std::size_t index = 0; index < expected.size(); ++index)
            {
                expected[index] = static_cast<std::uint8_t>(
                    (index * 53u + seed * 23u + (index >> 4u)) & 0xffu);
            }

            DeviceContiguousProjection source(
                *source_backend, source_device, projection_bytes);
            DeviceContiguousProjection destination(
                *destination_backend, destination_device, projection_bytes);
            IWorkerGPUContext &source_context =
                GPUDeviceContextPool::instance().getContext(source_device);
            IWorkerGPUContext &destination_context =
                GPUDeviceContextPool::instance().getContext(destination_device);
            const std::string identity =
                source_device.to_string() + "_to_" +
                destination_device.to_string() + "_" + precision_name;
            void *producer_stream =
                source_context.getOrCreateAuxiliaryStream(
                    "heterogeneous_float_test_producer:" + identity);
            void *source_inference_stream =
                source_context.getOrCreateAuxiliaryStream(
                    "heterogeneous_float_test_inference:" + identity);
            void *destination_inference_stream =
                destination_context.getOrCreateAuxiliaryStream(
                    "heterogeneous_float_test_inference:" + identity);
            ASSERT_NE(producer_stream, nullptr);
            ASSERT_NE(source_inference_stream, nullptr);
            ASSERT_NE(destination_inference_stream, nullptr);
            ASSERT_TRUE(source.upload(expected, producer_stream));

            void *source_ready_event =
                source_backend->createEvent(source_device.gpu_ordinal());
            ASSERT_NE(source_ready_event, nullptr);
            ASSERT_TRUE(source_backend->recordEvent(
                source_ready_event,
                source_device.gpu_ordinal(),
                producer_stream));
            const auto setup_deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            ASSERT_TRUE(awaitSetupEvent(
                source_context, source_ready_event, setup_deadline));
            auto progress_epochs = makeRetainedRelayEpochs(
                source_device,
                destination_device,
                staging_bytes,
                identity + ":" + std::to_string(seed));
            const auto source_mapped_staging =
                TransferEngine::instance().allocateMappedHostTransferSlices(
                    staging_bytes, 2u, source_device);
            const auto destination_mapped_staging =
                TransferEngine::instance().allocateMappedHostTransferSlices(
                    staging_bytes, 2u, destination_device);

            {
                ExpertTierGpuBlobTransferLane lane({
                    .source_device = source_device,
                    .destination_device = destination_device,
                    .staging_capacity_bytes = staging_bytes,
                    .source_mapped_staging = {
                        source_mapped_staging[0],
                        source_mapped_staging[1],
                    },
                    .destination_mapped_staging = {
                        destination_mapped_staging[0],
                        destination_mapped_staging[1],
                    },
                    .source_progress_epoch = progress_epochs.source,
                    .destination_progress_epoch =
                        progress_epochs.destination,
                    .lane_name = "integration_float:" + identity,
                    .perf_device = identity,
                    .collect_timing_measurements = true,
                });
                std::string error;
                ASSERT_TRUE(lane.materialize(&error)) << error;
                ASSERT_TRUE(lane.startContiguous(
                    source.data(),
                    destination.data(),
                    source.bytes(),
                    ExpertTierSourceReadiness::publishedResidencyBank(
                        seed + 1u),
                    &error))
                    << error;
                ASSERT_EQ(
                    lane.progress(),
                    ExpertTierGpuBlobTransferProgress::Pending);
                CapturedProgressReplay source_progress_replay(
                    source_context,
                    progress_epochs.source,
                    identity + ":source");
                CapturedProgressReplay destination_progress_replay(
                    destination_context,
                    progress_epochs.destination,
                    identity + ":destination");

                /*
                 * These copies represent unrelated captured inference work.
                 * They must finish without either stream joining migration.
                 */
                DeviceCopyWitness source_witness(
                    *source_backend, source_device);
                DeviceCopyWitness destination_witness(
                    *destination_backend, destination_device);
                ASSERT_TRUE(source_witness.enqueue(source_inference_stream));
                ASSERT_TRUE(destination_witness.enqueue(
                    destination_inference_stream));
                void *source_inference_event =
                    source_backend->createEvent(source_device.gpu_ordinal());
                void *destination_inference_event =
                    destination_backend->createEvent(
                        destination_device.gpu_ordinal());
                ASSERT_NE(source_inference_event, nullptr);
                ASSERT_NE(destination_inference_event, nullptr);
                ASSERT_TRUE(source_backend->recordEvent(
                    source_inference_event,
                    source_device.gpu_ordinal(),
                    source_inference_stream));
                ASSERT_TRUE(destination_backend->recordEvent(
                    destination_inference_event,
                    destination_device.gpu_ordinal(),
                    destination_inference_stream));

                bool source_inference_ready = false;
                bool destination_inference_ready = false;
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::seconds(10);
                while ((!source_inference_ready ||
                        !destination_inference_ready) &&
                       std::chrono::steady_clock::now() < deadline)
                {
                    ASSERT_TRUE(source_context.queryEventChecked(
                        source_inference_event, source_inference_ready));
                    ASSERT_TRUE(destination_context.queryEventChecked(
                        destination_inference_event,
                        destination_inference_ready));
                    std::this_thread::yield();
                }
                EXPECT_TRUE(source_inference_ready);
                EXPECT_TRUE(destination_inference_ready);
                EXPECT_EQ(
                    lane.progress(),
                    ExpertTierGpuBlobTransferProgress::Pending)
                    << "Inference work must not host-pump floating migration";

                while (lane.progress() ==
                           ExpertTierGpuBlobTransferProgress::Pending &&
                       std::chrono::steady_clock::now() < deadline)
                {
                    ASSERT_TRUE(source_progress_replay.launch());
                    ASSERT_TRUE(destination_progress_replay.launch());
                    ASSERT_NE(
                        lane.poll(&error),
                        ExpertTierGpuBlobTransferProgress::Failed)
                        << error;
                    std::this_thread::yield();
                }
                ASSERT_EQ(
                    lane.progress(),
                    ExpertTierGpuBlobTransferProgress::Ready)
                    << error;
                ASSERT_TRUE(source_progress_replay.awaitIdle(deadline));
                ASSERT_TRUE(destination_progress_replay.awaitIdle(deadline));

                const std::size_t expected_chunks =
                    (projection_bytes + staging_bytes - 1) / staging_bytes;
                const auto stats = lane.stats();
                EXPECT_EQ(stats.transfers_started, 1u);
                EXPECT_EQ(stats.transfers_completed, 1u);
                EXPECT_EQ(stats.chunks_submitted, expected_chunks);
                EXPECT_EQ(stats.chunks_completed, expected_chunks);
                EXPECT_EQ(stats.bytes_submitted, projection_bytes);
                EXPECT_EQ(stats.host_relay_bytes, projection_bytes);
                EXPECT_EQ(stats.source_d2h_submissions, expected_chunks);
                EXPECT_EQ(
                    stats.source_progress_kernel_submissions,
                    expected_chunks);
                EXPECT_EQ(stats.destination_h2d_submissions, expected_chunks);
                EXPECT_EQ(
                    stats.destination_progress_kernel_submissions,
                    expected_chunks);
                EXPECT_EQ(stats.maximum_in_flight_chunks, 2u);
                EXPECT_EQ(stats.failed_transfers, 0u);
                EXPECT_EQ(stats.timing_measurement_failures, 0u);
                EXPECT_TRUE(stats.last_measurement.valid());
                EXPECT_EQ(stats.last_measurement.bytes, projection_bytes);
                EXPECT_EQ(stats.last_measurement.device_nanoseconds, 0u);
                EXPECT_GT(stats.last_measurement.host_nanoseconds, 0u);
                EXPECT_EQ(stats.inference_stream_waits, 0u);
                EXPECT_EQ(stats.blocking_synchronizations, 0u);

                source_backend->destroyEvent(
                    source_inference_event,
                    source_device.gpu_ordinal());
                destination_backend->destroyEvent(
                    destination_inference_event,
                    destination_device.gpu_ordinal());
            }

            std::vector<std::uint8_t> actual;
            ASSERT_TRUE(destination.download(
                destination_inference_stream, actual));
            EXPECT_EQ(actual, expected);
            source_backend->destroyEvent(
                source_ready_event, source_device.gpu_ordinal());
        }

        /**
         * @brief Drive the production remote GPU endpoints without an MPI copy.
         *
         * The source endpoint produces the exact pinned span that MPI would
         * send and the destination immediately consumes that immutable span as
         * though its persistent receive completed. Waiting for destination H2D
         * completion before acknowledging the source is stricter than the live
         * MPI ownership edge and makes reuse bugs deterministic in this test.
         *
         * @param source_device CUDA or ROCm source endpoint.
         * @param destination_device Opposite-backend destination endpoint.
         * @param seed Deterministic payload and identity seed.
         */
        void runRemoteGpuBlobEndpointDirection(
            DeviceId source_device,
            DeviceId destination_device,
            std::uint32_t seed)
        {
            IBackend *source_backend = getBackendFor(source_device);
            IBackend *destination_backend = getBackendFor(destination_device);
            ASSERT_NE(source_backend, nullptr);
            ASSERT_NE(destination_backend, nullptr);

            constexpr int n = 130;
            constexpr int k = 96;
            constexpr std::size_t block_count =
                static_cast<std::size_t>(n) * (k / 32);
            std::vector<std::uint8_t> expected_payload(block_count * 16);
            std::vector<std::uint16_t> expected_scales(block_count);
            std::vector<std::uint16_t> expected_mins(block_count);
            for (std::size_t index = 0; index < expected_payload.size(); ++index)
            {
                expected_payload[index] = static_cast<std::uint8_t>(
                    (index * 29u + seed * 7u + 5u) & 0xffu);
            }
            for (std::size_t index = 0; index < block_count; ++index)
            {
                expected_scales[index] = static_cast<std::uint16_t>(
                    0x2800u + ((index * 11u + seed) & 0x03ffu));
                expected_mins[index] = static_cast<std::uint16_t>(
                    0xa400u + ((index * 17u + seed) & 0x03ffu));
            }

            DevicePackedProjection source(
                *source_backend, source_device, n, k);
            DevicePackedProjection destination(
                *destination_backend, destination_device, n, k);
            IWorkerGPUContext &source_context =
                GPUDeviceContextPool::instance().getContext(source_device);
            IWorkerGPUContext &destination_context =
                GPUDeviceContextPool::instance().getContext(destination_device);
            const std::string direction =
                source_device.to_string() + "_remote_to_" +
                destination_device.to_string();
            void *producer_stream = source_context.getOrCreateAuxiliaryStream(
                "remote_blob_test_producer:" + direction);
            void *observation_stream =
                destination_context.getOrCreateAuxiliaryStream(
                    "remote_blob_test_observer:" + direction);
            ASSERT_NE(producer_stream, nullptr);
            ASSERT_NE(observation_stream, nullptr);
            ASSERT_TRUE(source.upload(
                expected_payload,
                expected_scales,
                expected_mins,
                producer_stream));
            void *source_ready_event =
                source_backend->createEvent(source_device.gpu_ordinal());
            ASSERT_NE(source_ready_event, nullptr);
            ASSERT_TRUE(source_backend->recordEvent(
                source_ready_event,
                source_device.gpu_ordinal(),
                producer_stream));

            const MoEOverlayRemoteProjectionIdentity identity{
                .expected_epoch = 1000u + seed,
                .candidate_epoch = 1001u + seed,
                .execution_fingerprint = {
                    .low = 0xabc00000u + seed,
                    .high = 0xdef00000u + seed,
                },
                .migration_index = seed,
                .layer_idx = 4,
                .expert_id = 11,
                .projection = ExpertTierWeightProjection::Up,
                .source_participant = 0,
                .destination_participant = 1,
                .source_world_rank = 0,
                .destination_world_rank = 1,
                .source_device = source_device,
                .destination_device = destination_device,
            };
            const NativeVnniSourceIdentity source_identity{
                .codebook_id = 5,
                .is_superblock = false,
                .present = true,
            };
            const auto manifest = makeMoEOverlayRemoteGpuProjectionManifest(
                identity,
                source.descriptor(),
                source_identity,
                /*maximum_chunk_bytes=*/1024);
            ASSERT_TRUE(manifest.valid());

            auto source_lane = std::make_shared<
                MoEOverlayGpuRemoteProjectionLane>(
                MoEOverlayGpuRemoteProjectionLane::Config{
                    .device = source_device,
                    .staging = TransferEngine::instance()
                                   .allocatePersistentTransferStagingSlices(
                                       manifest.maximum_chunk_bytes,
                                       1u,
                                       source_device)
                                   .front(),
                    .execution = TransferEngine::instance()
                                     .allocatePersistentTransferExecutionLanes(
                                         1u,
                                         source_device,
                                         "remote_blob_source:" + direction)
                                     .front(),
                    .lane_name = "remote_blob_source:" + direction,
                    .perf_device = source_device.to_string(),
                });
            auto destination_lane = std::make_shared<
                MoEOverlayGpuRemoteProjectionLane>(
                MoEOverlayGpuRemoteProjectionLane::Config{
                    .device = destination_device,
                    .staging = TransferEngine::instance()
                                   .allocatePersistentTransferStagingSlices(
                                       manifest.maximum_chunk_bytes,
                                       1u,
                                       destination_device)
                                   .front(),
                    .execution = TransferEngine::instance()
                                     .allocatePersistentTransferExecutionLanes(
                                         1u,
                                         destination_device,
                                         "remote_blob_destination:" + direction)
                                     .front(),
                    .lane_name = "remote_blob_destination:" + direction,
                    .perf_device = destination_device.to_string(),
                });
            std::string error;
            ASSERT_TRUE(source_lane->materialize(&error)) << error;
            ASSERT_TRUE(destination_lane->materialize(&error)) << error;

            auto source_lifetime = std::make_shared<int>(1);
            auto destination_lifetime = std::make_shared<int>(2);
            {
                MoEOverlayGpuRemoteProjectionSource source_endpoint(
                    manifest,
                    source_lane,
                    source.descriptor(),
                    ExpertTierSourceReadiness::producerEvent(
                        source_ready_event),
                    source_lifetime);
                const auto destination_descriptor = destination.descriptor();
                MoEOverlayGpuRemoteProjectionDestination
                    destination_endpoint(
                        identity,
                        destination_lane,
                        [=](
                            const MoEOverlayRemoteProjectionManifest &received,
                            MoEOverlayGpuRemoteProjectionDestinationBinding *binding,
                            std::string *factory_error) -> bool
                        {
                            if (!binding || received != manifest)
                            {
                                if (factory_error)
                                    *factory_error =
                                        "Heterogeneous remote blob factory received a different manifest";
                                return false;
                            }
                            std::shared_ptr<ITensorGemm> engine;
                            if (destination_device.is_cuda())
                            {
                                engine = std::make_shared<
                                    cuda::CUDAQuantisedGemmKernel>(
                                    n,
                                    k,
                                    destination_device.gpu_ordinal(),
                                    destination_descriptor.ptrs.d_vnni,
                                    static_cast<std::uint16_t *>(
                                        destination_descriptor.ptrs.d_scales),
                                    static_cast<std::uint16_t *>(
                                        destination_descriptor.ptrs.d_mins),
                                    static_cast<std::uint32_t *>(
                                        destination_descriptor.ptrs.d_emins),
                                    destination_descriptor.codebook_id,
                                    destination_descriptor.blocks_per_row,
                                    destination_lifetime,
                                    source_identity);
                            }
                            else if (destination_device.is_rocm())
                            {
                                engine = std::make_shared<
                                    rocm::ROCmQuantisedGemmKernel>(
                                    n,
                                    k,
                                    destination_device.gpu_ordinal(),
                                    destination_descriptor.ptrs.d_vnni,
                                    destination_descriptor.ptrs.d_scales,
                                    destination_descriptor.ptrs.d_mins,
                                    destination_descriptor.ptrs.d_emins,
                                    destination_descriptor.codebook_id,
                                    destination_descriptor.blocks_per_row,
                                    destination_lifetime,
                                    source_identity);
                            }
                            if (!engine)
                            {
                                if (factory_error)
                                    *factory_error =
                                        "Heterogeneous remote blob factory has no destination backend";
                                return false;
                            }
                            *binding = {
                                .descriptor = destination_descriptor,
                                .engine = std::move(engine),
                            };
                            if (factory_error)
                                factory_error->clear();
                            return true;
                        },
                        destination_lifetime);
                ASSERT_TRUE(destination_endpoint.beginManifest(
                    manifest, &error))
                    << error;

                std::uint64_t chunks = 0;
                while (!destination_endpoint.complete())
                {
                    MoEOverlayRemoteProjectionChunkView chunk;
                    auto source_progress = source_endpoint.pollNextChunk(
                        &chunk, &error);
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::seconds(10);
                    while (source_progress ==
                               MoEOverlayResidencyWaveProgress::Pending &&
                           std::chrono::steady_clock::now() < deadline)
                    {
                        source_progress = source_endpoint.pollNextChunk(
                            &chunk, &error);
                        std::this_thread::yield();
                    }
                    ASSERT_EQ(
                        source_progress,
                        MoEOverlayResidencyWaveProgress::Ready)
                        << error;

                    auto destination_progress =
                        destination_endpoint.beginChunk(
                            chunk.header, chunk.payload, &error);
                    while (destination_progress ==
                               MoEOverlayResidencyWaveProgress::Pending &&
                           std::chrono::steady_clock::now() < deadline)
                    {
                        destination_progress =
                            destination_endpoint.pollChunk(&error);
                        std::this_thread::yield();
                    }
                    ASSERT_EQ(
                        destination_progress,
                        MoEOverlayResidencyWaveProgress::Ready)
                        << error;
                    ASSERT_TRUE(source_endpoint.acknowledgeChunkSent(
                        chunk.header, &error))
                        << error;
                    ++chunks;
                }
                ASSERT_TRUE(destination_endpoint.publishFinal(&error))
                    << error;
                ASSERT_NE(destination_endpoint.preparedEngine(), nullptr);
                EXPECT_GT(chunks, 4u)
                    << "All separated regions should span multiple network chunks";

                const auto source_stats = source_lane->stats();
                const auto destination_stats = destination_lane->stats();
                EXPECT_EQ(source_stats.gpu_blob_read_chunks, chunks);
                EXPECT_EQ(destination_stats.gpu_blob_write_chunks, chunks);
                EXPECT_EQ(source_stats.device_to_host_submissions, chunks);
                EXPECT_EQ(destination_stats.host_to_device_submissions, chunks);
                EXPECT_EQ(source_stats.inference_stream_waits, 0u);
                EXPECT_EQ(destination_stats.inference_stream_waits, 0u);
                EXPECT_EQ(source_stats.blocking_synchronizations, 0u);
                EXPECT_EQ(destination_stats.blocking_synchronizations, 0u);
            }

            std::vector<std::uint8_t> actual_payload;
            std::vector<std::uint16_t> actual_scales;
            std::vector<std::uint16_t> actual_mins;
            ASSERT_TRUE(destination.download(
                observation_stream,
                actual_payload,
                actual_scales,
                actual_mins));
            EXPECT_EQ(actual_payload, expected_payload);
            EXPECT_EQ(actual_scales, expected_scales);
            EXPECT_EQ(actual_mins, expected_mins);
            source_backend->destroyEvent(
                source_ready_event,
                source_device.gpu_ordinal());
        }

        /**
         * @brief Construct the real backend floating GEMM alias for a test slot.
         * @param device Exact CUDA or ROCm destination.
         * @param descriptor Complete raw floating weight view.
         * @param lifetime Slot lifetime retained by the alias.
         * @return Executable destination engine, or null for an invalid backend.
         */
        std::shared_ptr<ITensorGemm> makeFloatingDestinationEngine(
            DeviceId device,
            const ContiguousFloatingPointWeightDescriptor &descriptor,
            std::shared_ptr<void> lifetime)
        {
            if (device.is_cuda())
            {
                cuda::CUDAFloatingPointGemmKernel::Precision precision;
                switch (descriptor.type)
                {
                case TensorType::FP16:
                    precision = cuda::CUDAFloatingPointGemmKernel::Precision::FP16;
                    break;
                case TensorType::BF16:
                    precision = cuda::CUDAFloatingPointGemmKernel::Precision::BF16;
                    break;
                case TensorType::FP32:
                    precision = cuda::CUDAFloatingPointGemmKernel::Precision::FP32;
                    break;
                default:
                    return nullptr;
                }
                return std::make_shared<cuda::CUDAFloatingPointGemmKernel>(
                    descriptor.data,
                    descriptor.n,
                    descriptor.k,
                    device.cuda_ordinal(),
                    precision,
                    std::move(lifetime));
            }
            if (device.is_rocm())
            {
                rocm::ROCmFloatingPointGemmKernel::Precision precision;
                switch (descriptor.type)
                {
                case TensorType::FP16:
                    precision = rocm::ROCmFloatingPointGemmKernel::Precision::FP16;
                    break;
                case TensorType::BF16:
                    precision = rocm::ROCmFloatingPointGemmKernel::Precision::BF16;
                    break;
                case TensorType::FP32:
                    precision = rocm::ROCmFloatingPointGemmKernel::Precision::FP32;
                    break;
                default:
                    return nullptr;
                }
                return std::make_shared<rocm::ROCmFloatingPointGemmKernel>(
                    descriptor.data,
                    descriptor.n,
                    descriptor.k,
                    device.rocm_ordinal(),
                    precision,
                    std::move(lifetime));
            }
            return nullptr;
        }

        /**
         * @brief Drive one raw floating projection through both remote endpoints.
         * @param source_device CUDA or ROCm source endpoint.
         * @param destination_device Opposite-backend destination endpoint.
         * @param type FP16, BF16, or FP32 storage precision.
         * @param seed Deterministic byte-pattern and transaction seed.
         */
        void runRemoteFloatingEndpointDirection(
            DeviceId source_device,
            DeviceId destination_device,
            TensorType type,
            std::uint32_t seed)
        {
            IBackend *source_backend = getBackendFor(source_device);
            IBackend *destination_backend = getBackendFor(destination_device);
            ASSERT_NE(source_backend, nullptr);
            ASSERT_NE(destination_backend, nullptr);

            constexpr int n = 131;
            constexpr int k = 97;
            const auto format = ExpertWeightFormat::floating(type);
            ASSERT_TRUE(format.valid());
            const std::size_t projection_bytes =
                static_cast<std::size_t>(n) * k *
                format.floatingElementBytes();
            std::vector<std::uint8_t> expected(projection_bytes);
            for (std::size_t index = 0; index < expected.size(); ++index)
            {
                expected[index] = static_cast<std::uint8_t>(
                    (index * 61u + seed * 19u + (index >> 3u)) & 0xffu);
            }

            DeviceContiguousProjection source(
                *source_backend, source_device, projection_bytes);
            DeviceContiguousProjection destination(
                *destination_backend,
                destination_device,
                projection_bytes);
            auto &source_context =
                GPUDeviceContextPool::instance().getContext(source_device);
            auto &destination_context =
                GPUDeviceContextPool::instance().getContext(
                    destination_device);
            const std::string direction =
                source_device.to_string() + "_remote_float_to_" +
                destination_device.to_string() + "_" +
                tensorTypeName(type);
            void *producer_stream =
                source_context.getOrCreateAuxiliaryStream(
                    "remote_float_test_producer:" + direction);
            void *observation_stream =
                destination_context.getOrCreateAuxiliaryStream(
                    "remote_float_test_observer:" + direction);
            ASSERT_NE(producer_stream, nullptr);
            ASSERT_NE(observation_stream, nullptr);
            ASSERT_TRUE(source.upload(expected, producer_stream));
            void *source_ready_event =
                source_backend->createEvent(source_device.gpu_ordinal());
            ASSERT_NE(source_ready_event, nullptr);
            ASSERT_TRUE(source_backend->recordEvent(
                source_ready_event,
                source_device.gpu_ordinal(),
                producer_stream));

            const MoEOverlayRemoteProjectionIdentity identity{
                .expected_epoch = 2000u + seed,
                .candidate_epoch = 2001u + seed,
                .execution_fingerprint = {
                    .low = 0xf1000000u + seed,
                    .high = 0xf2000000u + seed,
                },
                .migration_index = seed,
                .layer_idx = 41,
                .expert_id = 17,
                .projection = ExpertTierWeightProjection::Down,
                .source_participant = 0,
                .destination_participant = 1,
                .source_world_rank = 0,
                .destination_world_rank = 1,
                .source_device = source_device,
                .destination_device = destination_device,
            };
            const ContiguousFloatingPointWeightDescriptor source_descriptor{
                .data = source.data(),
                .type = type,
                .n = n,
                .k = k,
                .bytes = projection_bytes,
            };
            const ContiguousFloatingPointWeightDescriptor
                destination_descriptor{
                    .data = destination.data(),
                    .type = type,
                    .n = n,
                    .k = k,
                    .bytes = projection_bytes,
                };
            const auto manifest =
                makeMoEOverlayRemoteFloatingProjectionManifest(
                    identity,
                    source_descriptor,
                    /*maximum_chunk_bytes=*/1024u);
            ASSERT_TRUE(manifest.valid());

            auto source_lane = std::make_shared<
                MoEOverlayGpuRemoteProjectionLane>(
                    MoEOverlayGpuRemoteProjectionLane::Config{
                        .device = source_device,
                        .staging = TransferEngine::instance()
                                       .allocatePersistentTransferStagingSlices(
                                           manifest.maximum_chunk_bytes,
                                           1u,
                                           source_device)
                                       .front(),
                        .execution = TransferEngine::instance()
                                         .allocatePersistentTransferExecutionLanes(
                                             1u,
                                             source_device,
                                             "remote_float_source:" + direction)
                                         .front(),
                        .lane_name = "remote_float_source:" + direction,
                        .perf_device = source_device.to_string(),
                    });
            auto destination_lane = std::make_shared<
                MoEOverlayGpuRemoteProjectionLane>(
                    MoEOverlayGpuRemoteProjectionLane::Config{
                        .device = destination_device,
                        .staging = TransferEngine::instance()
                                       .allocatePersistentTransferStagingSlices(
                                           manifest.maximum_chunk_bytes,
                                           1u,
                                           destination_device)
                                       .front(),
                        .execution = TransferEngine::instance()
                                         .allocatePersistentTransferExecutionLanes(
                                             1u,
                                             destination_device,
                                             "remote_float_destination:" + direction)
                                         .front(),
                        .lane_name = "remote_float_destination:" + direction,
                        .perf_device = destination_device.to_string(),
                    });
            std::string error;
            ASSERT_TRUE(source_lane->materialize(&error)) << error;
            ASSERT_TRUE(destination_lane->materialize(&error)) << error;

            auto source_lifetime = std::make_shared<int>(3);
            auto destination_lifetime = std::make_shared<int>(4);
            {
                MoEOverlayGpuRemoteProjectionSource source_endpoint(
                    manifest,
                    source_lane,
                    source_descriptor,
                    ExpertTierSourceReadiness::producerEvent(
                        source_ready_event),
                    source_lifetime);
                MoEOverlayGpuRemoteProjectionDestination
                    destination_endpoint(
                        identity,
                        destination_lane,
                        [=](
                            const MoEOverlayRemoteProjectionManifest &received,
                            MoEOverlayGpuRemoteProjectionDestinationBinding *binding,
                            std::string *factory_error) -> bool
                        {
                            if (!binding || received != manifest)
                            {
                                if (factory_error)
                                    *factory_error =
                                        "Remote floating factory received a different manifest";
                                return false;
                            }
                            auto engine = makeFloatingDestinationEngine(
                                destination_device,
                                destination_descriptor,
                                destination_lifetime);
                            if (!engine)
                            {
                                if (factory_error)
                                    *factory_error =
                                        "Remote floating factory has no destination backend";
                                return false;
                            }
                            *binding = {
                                .floating_descriptor =
                                    destination_descriptor,
                                .engine = std::move(engine),
                            };
                            if (factory_error)
                                factory_error->clear();
                            return true;
                        },
                        destination_lifetime);
                ASSERT_TRUE(destination_endpoint.beginManifest(
                    manifest, &error)) << error;

                std::uint64_t chunks = 0;
                while (!destination_endpoint.complete())
                {
                    MoEOverlayRemoteProjectionChunkView chunk;
                    auto source_progress = source_endpoint.pollNextChunk(
                        &chunk, &error);
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::seconds(10);
                    while (source_progress ==
                               MoEOverlayResidencyWaveProgress::Pending &&
                           std::chrono::steady_clock::now() < deadline)
                    {
                        source_progress = source_endpoint.pollNextChunk(
                            &chunk, &error);
                        std::this_thread::yield();
                    }
                    ASSERT_EQ(
                        source_progress,
                        MoEOverlayResidencyWaveProgress::Ready) << error;

                    auto destination_progress =
                        destination_endpoint.beginChunk(
                            chunk.header, chunk.payload, &error);
                    while (destination_progress ==
                               MoEOverlayResidencyWaveProgress::Pending &&
                           std::chrono::steady_clock::now() < deadline)
                    {
                        destination_progress =
                            destination_endpoint.pollChunk(&error);
                        std::this_thread::yield();
                    }
                    ASSERT_EQ(
                        destination_progress,
                        MoEOverlayResidencyWaveProgress::Ready) << error;
                    ASSERT_TRUE(source_endpoint.acknowledgeChunkSent(
                        chunk.header, &error)) << error;
                    ++chunks;
                }
                ASSERT_TRUE(destination_endpoint.publishFinal(&error))
                    << error;
                ASSERT_NE(destination_endpoint.preparedEngine(), nullptr);
                EXPECT_GT(chunks, 8u);

                const auto source_stats = source_lane->stats();
                const auto destination_stats = destination_lane->stats();
                EXPECT_EQ(source_stats.gpu_blob_read_chunks, chunks);
                EXPECT_EQ(destination_stats.gpu_blob_write_chunks, chunks);
                EXPECT_EQ(source_stats.inference_stream_waits, 0u);
                EXPECT_EQ(destination_stats.inference_stream_waits, 0u);
                EXPECT_EQ(source_stats.blocking_synchronizations, 0u);
                EXPECT_EQ(destination_stats.blocking_synchronizations, 0u);
            }

            std::vector<std::uint8_t> actual;
            ASSERT_TRUE(destination.download(observation_stream, actual));
            EXPECT_EQ(actual, expected);
            source_backend->destroyEvent(
                source_ready_event,
                source_device.gpu_ordinal());
        }

        TEST(
            ExpertTierGpuBlobTransferIntegration,
            CUDAAndROCmRelayBothDirectionsByteExactlyWithoutInferenceWaits)
        {
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
            GTEST_SKIP() << "CUDA and ROCm are both required";
#else
            IBackend *cuda = getCUDABackend();
            IBackend *rocm = getROCmBackend();
            if (!cuda || !rocm ||
                cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
            {
                GTEST_SKIP() << "One CUDA and one ROCm device are required";
            }

            ScopedHeterogeneousBlobPerfStats perf_stats;
            runHeterogeneousBlobDirection(
                DeviceId::cuda(0),
                DeviceId::rocm(0),
                991u);
            runHeterogeneousBlobDirection(
                DeviceId::rocm(0),
                DeviceId::cuda(0),
                997u);

            double completed = 0.0;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"moe_overlay_residency"}))
            {
                if (record.kind == PerfStatRecord::Kind::Counter &&
                    record.name ==
                        "gpu_host_relay_transfers_completed")
                {
                    EXPECT_EQ(
                        record.tags.at("source_transport"),
                        "retained_mapped_progress_epoch");
                    EXPECT_EQ(
                        record.tags.at("stream_class"),
                        "latency_critical");
                    completed += record.value;
                }
            }
            EXPECT_EQ(completed, 2.0)
                << "PerfStats must prove both cross-vendor movements occurred";
#endif
        }

        TEST(
            ExpertTierGpuBlobTransferIntegration,
            CUDAWithoutPeerAccessRelaysBothDirectionsByteExactlyInBackground)
        {
#if !defined(HAVE_CUDA)
            GTEST_SKIP() << "CUDA is required";
#else
            IBackend *cuda = getCUDABackend();
            if (!cuda || cuda->deviceCount() < 2)
                GTEST_SKIP() << "Two CUDA devices are required";

            DeviceManager &devices = DeviceManager::instance();
            devices.initialize(-1, false);
            const auto zero_reads_one = devices.peerAccessAvailable(
                DeviceId::cuda(0), DeviceId::cuda(1));
            const auto one_reads_zero = devices.peerAccessAvailable(
                DeviceId::cuda(1), DeviceId::cuda(0));
            ASSERT_TRUE(zero_reads_one.has_value());
            ASSERT_TRUE(one_reads_zero.has_value());
            if (*zero_reads_one || *one_reads_zero)
            {
                GTEST_SKIP()
                    << "This certificate requires a CUDA pair with no directed "
                       "peer access; native peer lanes are authoritative here";
            }

            ScopedHeterogeneousBlobPerfStats perf_stats;
            runHeterogeneousBlobDirection(
                DeviceId::cuda(0),
                DeviceId::cuda(1),
                1009u,
                ExpertTierGpuBlobRelayKind::SameBackendWithoutPeerAccess);
            runHeterogeneousBlobDirection(
                DeviceId::cuda(1),
                DeviceId::cuda(0),
                1013u,
                ExpertTierGpuBlobRelayKind::SameBackendWithoutPeerAccess);

            double completed = 0.0;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"moe_overlay_residency"}))
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.name != "gpu_host_relay_transfers_completed")
                {
                    continue;
                }
                const auto relay_kind = record.tags.find("relay_kind");
                const auto background = record.tags.find("background");
                const auto blocking = record.tags.find("blocking");
                const auto source_transport =
                    record.tags.find("source_transport");
                const auto stream_class =
                    record.tags.find("stream_class");
                if (relay_kind != record.tags.end() &&
                    relay_kind->second == "same_backend_no_peer" &&
                    background != record.tags.end() &&
                    background->second == "true" &&
                    blocking != record.tags.end() &&
                    blocking->second == "false" &&
                    source_transport != record.tags.end() &&
                    source_transport->second ==
                        "retained_mapped_progress_epoch" &&
                    stream_class != record.tags.end() &&
                    stream_class->second == "latency_critical")
                {
                    completed += record.value;
                }
            }
            EXPECT_EQ(completed, 2.0)
                << "PerfStats must prove both no-P2P CUDA movements used the "
                   "non-blocking background host relay";
#endif
        }

        TEST(
            ExpertTierGpuBlobTransferIntegration,
            FloatingProjectionRelayCoversAllPrecisionsAndBothDirections)
        {
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
            GTEST_SKIP() << "CUDA and ROCm are both required";
#else
            IBackend *cuda = getCUDABackend();
            IBackend *rocm = getROCmBackend();
            if (!cuda || !rocm || cuda->deviceCount() < 1 ||
                rocm->deviceCount() < 1)
            {
                GTEST_SKIP() << "One CUDA and one ROCm device are required";
            }

            ScopedHeterogeneousBlobPerfStats perf_stats;
            struct FloatingCase
            {
                const char *name;
                std::size_t element_bytes;
                std::uint32_t seed;
            };
            constexpr std::array<FloatingCase, 3> cases{{
                {"fp16", 2, 0xF016u},
                {"bf16", 2, 0xBF16u},
                {"fp32", 4, 0xF032u},
            }};
            for (const FloatingCase &test_case : cases)
            {
                runHeterogeneousContiguousDirection(
                    DeviceId::cuda(0),
                    DeviceId::rocm(0),
                    test_case.name,
                    test_case.element_bytes,
                    test_case.seed);
                runHeterogeneousContiguousDirection(
                    DeviceId::rocm(0),
                    DeviceId::cuda(0),
                    test_case.name,
                    test_case.element_bytes,
                    test_case.seed + 101u);
            }

            double completed = 0.0;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"moe_overlay_residency"}))
            {
                if (record.kind == PerfStatRecord::Kind::Counter &&
                    record.name ==
                        "gpu_host_relay_transfers_completed")
                {
                    EXPECT_EQ(record.tags.at("background"), "true");
                    EXPECT_EQ(record.tags.at("blocking"), "false");
                    EXPECT_EQ(
                        record.tags.at("source_transport"),
                        "retained_mapped_progress_epoch");
                    EXPECT_EQ(
                        record.tags.at("stream_class"),
                        "latency_critical");
                    completed += record.value;
                }
            }
            EXPECT_EQ(completed, 6.0)
                << "PerfStats must prove every precision moved both ways";
#endif
        }

        TEST(
            ExpertTierGpuBlobTransferIntegration,
            RemoteEndpointsPreserveCUDAAndROCmBlobsBothDirections)
        {
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
            GTEST_SKIP() << "CUDA and ROCm are both required";
#else
            IBackend *cuda = getCUDABackend();
            IBackend *rocm = getROCmBackend();
            if (!cuda || !rocm || cuda->deviceCount() < 1 ||
                rocm->deviceCount() < 1)
            {
                GTEST_SKIP() << "One CUDA and one ROCm device are required";
            }

            ScopedHeterogeneousBlobPerfStats perf_stats;
            runRemoteGpuBlobEndpointDirection(
                DeviceId::cuda(0), DeviceId::rocm(0), 1201u);
            runRemoteGpuBlobEndpointDirection(
                DeviceId::rocm(0), DeviceId::cuda(0), 1207u);

            double completed_chunks = 0.0;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"moe_overlay_residency"}))
            {
                if (record.kind == PerfStatRecord::Kind::Counter &&
                    record.name == "remote_gpu_chunks_completed")
                {
                    EXPECT_EQ(record.tags.at("background"), "true");
                    EXPECT_EQ(record.tags.at("blocking"), "false");
                    completed_chunks += record.value;
                }
            }
            EXPECT_GT(completed_chunks, 16.0)
                << "PerfStats must prove both remote cross-vendor blob paths ran";
#endif
        }

        TEST(
            ExpertTierGpuBlobTransferIntegration,
            RemoteEndpointsPreserveEveryFloatingPrecisionBothDirections)
        {
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
            GTEST_SKIP() << "CUDA and ROCm are both required";
#else
            IBackend *cuda = getCUDABackend();
            IBackend *rocm = getROCmBackend();
            if (!cuda || !rocm || cuda->deviceCount() < 1 ||
                rocm->deviceCount() < 1)
            {
                GTEST_SKIP() << "One CUDA and one ROCm device are required";
            }

            ScopedHeterogeneousBlobPerfStats perf_stats;
            constexpr std::array<TensorType, 3> types{
                TensorType::FP16,
                TensorType::BF16,
                TensorType::FP32,
            };
            std::uint32_t seed = 2201u;
            for (const auto type : types)
            {
                runRemoteFloatingEndpointDirection(
                    DeviceId::cuda(0), DeviceId::rocm(0), type, seed++);
                runRemoteFloatingEndpointDirection(
                    DeviceId::rocm(0), DeviceId::cuda(0), type, seed++);
            }

            double completed_chunks = 0.0;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"moe_overlay_residency"}))
            {
                if (record.kind == PerfStatRecord::Kind::Counter &&
                    record.name == "remote_gpu_chunks_completed")
                {
                    EXPECT_EQ(record.tags.at("background"), "true");
                    EXPECT_EQ(record.tags.at("blocking"), "false");
                    completed_chunks += record.value;
                }
            }
            EXPECT_GT(completed_chunks, 100.0)
                << "PerfStats must prove every floating precision moved both ways";
#endif
        }
    } // namespace
} // namespace llaminar2
