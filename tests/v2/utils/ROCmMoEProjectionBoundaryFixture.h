/**
 * @file ROCmMoEProjectionBoundaryFixture.h
 * @brief Persistent production-kernel fixture for exact expert phase boundaries.
 *
 * Weights use the real loader/repacker. Column slices copy native packed bytes,
 * including every scale plane and source identity; no dequantize/requantize is
 * permitted. Only setup and result collection synchronize. Recorded work calls
 * the same launch bridge as ROCmMoEKernel, with no alternative device arithmetic.
 */
#pragma once

#include "GpuPreparedGemmHarness.h"
#include "QuantizedVerifierFormats.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/moe/MoEExpertProjectionOwnership.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.h"
#include "transfer/TransferEngine.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    /** @brief Explicit diagnostic geometry; not a production placement policy. */
    struct ProjectionFixturePlacement
    {
        DeviceId device = DeviceId::rocm(0);
        int experts = 3;
        int top_k = 2;
    };

    /** @brief Select the complete oracle grouping or one producer's owned rows. */
    enum class ProjectionGroupingCoverage { AllRoutes, OwnedRoutes };

    /** @brief Replay input for the fixture's synthetic authoritative ownership. */
    struct ProjectionFixtureGrouping
    {
        int participants = 1;
        int participant = 0;
        ProjectionGroupingCoverage coverage = ProjectionGroupingCoverage::AllRoutes;
    };

    /** @brief Preserve independent source formats for the two projection roles. */
    struct ProjectionFixtureFormats
    {
        const QuantizedVerifierFormatCase &gate_up;
        const QuantizedVerifierFormatCase &down;
    };
    /** @brief Raise the exact runtime failure outside the measured transaction. */
    inline void checkProjectionHip(hipError_t result, const char *operation)
    {
        if (result != hipSuccess)
            throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
    }

    /** @brief Retain one immutable complete graph through explicit-stream replay. */
    class ProjectionBoundaryGraph final
    {
    public:
        /** @brief Capture exactly the supplied production operations. */
        template <typename Body>
        ProjectionBoundaryGraph(hipStream_t stream, Body &&body) : stream_(stream)
        {
            if (!stream) throw std::invalid_argument("projection graph needs an explicit stream");
            checkProjectionHip(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin projection capture");
            try { body(); }
            catch (...)
            {
                (void)hipStreamEndCapture(stream, &graph_);
                if (graph_) (void)hipGraphDestroy(graph_);
                throw;
            }
            checkProjectionHip(hipStreamEndCapture(stream, &graph_), "end projection capture");
            const auto result = hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0);
            if (result != hipSuccess)
            {
                (void)hipGraphDestroy(graph_);
                checkProjectionHip(result, "instantiate projection graph");
            }
        }

        /** @brief Join before retiring the executable's borrowed addresses. */
        ~ProjectionBoundaryGraph()
        {
            (void)hipStreamSynchronize(stream_);
            (void)hipGraphExecDestroy(executable_);
            (void)hipGraphDestroy(graph_);
        }
        ProjectionBoundaryGraph(const ProjectionBoundaryGraph &) = delete;
        ProjectionBoundaryGraph &operator=(const ProjectionBoundaryGraph &) = delete;

        /** @brief Submit only the retained graph; no host route or output access. */
        void replay() const { checkProjectionHip(hipGraphLaunch(executable_, stream_), "replay projection graph"); }

    private:
        hipStream_t stream_;
        hipGraph_t graph_ = nullptr;
        hipGraphExec_t executable_ = nullptr;
    };

    /**
     * @brief Own all descriptors, prepared weights and scratch for one geometry.
     *
     * This fixture can retain an oracle and candidate simultaneously; that is
     * diagnostic storage, not a production allocation plan. A model integration
     * must prove its own unchanged PhysicalMemoryAuthority bill of materials.
     */
    class ROCmMoEProjectionBoundaryFixture final
    {
    public:
        /** @brief Materialize fixed geometry and weights before recording any graph. */
        ROCmMoEProjectionBoundaryFixture(const QuantizedVerifierFormatCase &format,
            int model_width = 512, int expert_width = 256, int row_capacity = 65,
            ProjectionFixturePlacement placement = {})
            : ROCmMoEProjectionBoundaryFixture(ProjectionFixtureFormats{format, format},
                model_width, expert_width, row_capacity, placement) {}

        /** @brief Retain mixed source formats exactly as a real layer does. */
        ROCmMoEProjectionBoundaryFixture(ProjectionFixtureFormats formats,
            int model_width, int expert_width, int row_capacity,
            ProjectionFixturePlacement placement)
            : width(model_width), intermediate(expert_width), capacity(row_capacity),
              experts(placement.experts), top_k(placement.top_k), device(placement.device),
              formats_(formats)
        {
            if (!device.is_rocm() || experts < top_k || top_k < 1 || capacity < 1)
                throw std::invalid_argument("invalid projection fixture geometry");
            checkProjectionHip(hipSetDevice(device.ordinal), "select projection device");
            checkProjectionHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create projection stream");
            const int slots = capacity * top_k;
            const int max_dim = std::max(width, intermediate);
            hidden = allocate<float>(static_cast<size_t>(capacity) * width);
            counts = allocate<int>(experts);
            offsets = allocate<int>(experts + 1);
            tokens = allocate<int>(slots);
            inverse = allocate<int>(slots);
            expert_ids = allocate<int>(slots);
            route_owners = allocate<int>(slots);
            route_weights = allocate<float>(slots);
            directory = allocate<int>(MoEWorkspaceBuffers::rocmAdaptivePrefillDirectoryWords(slots, experts));
            input_q8 = allocate<int8_t>(static_cast<size_t>(slots) * max_dim);
            input_scales = allocate<float>(static_cast<size_t>(slots) * ((max_dim + 31) / 32));
            gate = allocate<float>(static_cast<size_t>(slots) * max_dim);
            up = allocate<float>(static_cast<size_t>(slots) * intermediate);
            swiglu = allocate<int8_t>(static_cast<size_t>(slots) * intermediate);
            swiglu_scales = allocate<float>(static_cast<size_t>(slots) * (intermediate / 32));
            output = allocate<float>(static_cast<size_t>(capacity) * width);
            std::vector<DeviceNativeVNNIMatrixDesc> gate_host, up_host;
            for (int expert = 0; expert < experts; ++expert)
            {
                gate_host.push_back(prepare(formats_.gate_up, intermediate, width, 1100 + expert));
                up_host.push_back(prepare(formats_.gate_up, intermediate, width, 1200 + expert));
                down_host_.push_back(prepare(formats_.down, width, intermediate, 1300 + expert));
            }
            gate_table = upload(gate_host);
            up_table = upload(up_host);
            down_table = upload(down_host_);
        }

        /** @brief Complete fixture work before releasing descriptor backing bytes. */
        ~ROCmMoEProjectionBoundaryFixture()
        {
            (void)hipStreamSynchronize(stream);
            buffers_.clear();
            prepared_.clear();
            weights_.clear();
            (void)hipStreamDestroy(stream);
        }
        ROCmMoEProjectionBoundaryFixture(const ROCmMoEProjectionBoundaryFixture &) = delete;
        ROCmMoEProjectionBoundaryFixture &operator=(const ROCmMoEProjectionBoundaryFixture &) = delete;

        /** @return Stable allocation tracked by the public transfer authority. */
        template <typename T> T *allocate(size_t count)
        {
            auto owner = TransferEngine::instance().allocateDeviceTransferBuffer(count * sizeof(T), device);
            if (!owner || !owner->isBound()) throw std::runtime_error("projection allocation failed");
            auto *pointer = static_cast<T *>(owner->mutableDeviceData());
            buffers_.push_back(std::move(owner));
            checkProjectionHip(hipMemsetAsync(pointer, 0, count * sizeof(T), stream), "initialize projection storage");
            return pointer;
        }

        /** @brief Upload immutable setup data, retaining its host lifetime until complete. */
        template <typename T> T *upload(const std::vector<T> &values)
        {
            T *destination = allocate<T>(values.size());
            publish(destination, values);
            return destination;
        }

        /** @brief Publish a new replay's input outside the timed/captured interval. */
        template <typename T> void publish(T *destination, const std::vector<T> &values)
        {
            checkProjectionHip(hipMemcpyAsync(destination, values.data(), values.size() * sizeof(T),
                hipMemcpyHostToDevice, stream), "publish projection input");
            checkProjectionHip(hipStreamSynchronize(stream), "join projection setup input");
        }

        /** @return Completed output bytes; never called inside capture or timing. */
        template <typename T> std::vector<T> download(const T *source, size_t count)
        {
            std::vector<T> result(count);
            if (count == 0) return result;
            checkProjectionHip(hipMemcpyAsync(result.data(), source, count * sizeof(T),
                hipMemcpyDeviceToHost, stream), "read projection output");
            checkProjectionHip(hipStreamSynchronize(stream), "join projection readback");
            return result;
        }

        /**
         * @brief Publish stable expert-major groups for adversarial original routes.
         *
         * Every live row has two distinct experts in changing router order.
         * Inactive inverse entries are -1 and stale scratch is not cleared, so
         * short/empty replays prove that old intermediate rows cannot leak out.
         */
        void reset(int physical_rows, int live_rows, int seed, ProjectionFixtureGrouping grouping = {})
        {
            if (grouping.participants < 1 || grouping.participant < 0 || grouping.participant >= grouping.participants ||
                physical_rows < 1 || physical_rows > capacity || live_rows < 0 || live_rows > physical_rows)
                throw std::invalid_argument("invalid projection replay grouping");
            std::vector<float> values(static_cast<size_t>(physical_rows) * width);
            for (size_t i = 0; i < values.size(); ++i)
                // Keep composed low-bit FFNs above the activation quantizer's
                // tiny-value floor; equality of two all-zero rows is too weak.
                values[i] = static_cast<float>(static_cast<int>((i * 37 + seed * 13) % 101) - 50) * 0.0037f;
            publish(hidden, values);
            std::vector<int> ids(physical_rows * top_k, -1), map(ids.size(), -1), group_tokens(ids.size(), -1);
            std::vector<int> owners(ids.size(), -1);
            std::vector<float> weights(ids.size(), 0.0f);
            std::vector<int> group_counts(experts, 0), group_offsets(experts + 1, 0);
            for (int row = 0; row < live_rows; ++row)
                for (int route = 0; route < top_k; ++route)
                {
                    const int slot = row * top_k + route;
                    ids[slot] = (row * top_k + route + seed) % experts;
                    owners[slot] = (ids[slot] + seed) % grouping.participants;
                    if (grouping.coverage == ProjectionGroupingCoverage::AllRoutes || owners[slot] == grouping.participant)
                        ++group_counts[ids[slot]];
                }
            for (int expert = 0; expert < experts; ++expert)
                group_offsets[expert + 1] = group_offsets[expert] + group_counts[expert];
            auto next = group_offsets;
            for (int slot = 0; slot < live_rows * top_k; ++slot)
            {
                if (grouping.coverage == ProjectionGroupingCoverage::OwnedRoutes && owners[slot] != grouping.participant)
                    continue;
                const int grouped = next[ids[slot]]++;
                map[slot] = grouped;
                group_tokens[grouped] = slot / top_k;
                weights[grouped] = (slot % top_k == 0) ? 0.625f : 0.375f;
            }
            publish(counts, group_counts);
            publish(offsets, group_offsets);
            publish(tokens, group_tokens);
            publish(inverse, map);
            publish(expert_ids, ids);
            publish(route_owners, owners);
            publish(route_weights, weights);
        }

        /** @brief Launch the production bridge with unowned phase bindings absent. */
        void launch(int rows, const MoEPrefillProjectionExecution &execution,
            const DeviceNativeVNNIMatrixDesc *selected_down = nullptr, float *selected_output = nullptr)
        {
            launchOnStream(stream, rows, execution, selected_down, selected_output);
        }

        /** @brief Bind the exact shared transaction stream, never a guessed producer. */
        void launchOnStream(hipStream_t execution_stream, int rows, const MoEPrefillProjectionExecution &execution,
            const DeviceNativeVNNIMatrixDesc *selected_down = nullptr, float *selected_output = nullptr,
            float *canonical_routes = nullptr)
        {
            if (!execution_stream) throw std::invalid_argument("projection launch needs an explicit stream");
            const bool producer = execution.executesGateUp();
            const bool consumer = execution.executesDown();
            if (!rocmMoE_grouped_prefill_pipeline(
                    producer ? hidden : nullptr, nullptr, nullptr,
                    producer ? gate_table : nullptr, producer ? up_table : nullptr,
                    consumer ? (selected_down ? selected_down : down_table) : nullptr,
                    counts, offsets, producer ? tokens : nullptr,
                    consumer ? inverse : nullptr, consumer ? expert_ids : nullptr,
                    consumer ? route_weights : nullptr, directory,
                    producer ? input_q8 : nullptr, producer ? input_scales : nullptr,
                    gate, producer ? up : nullptr, swiglu, swiglu_scales,
                    consumer && !canonical_routes ? (selected_output ? selected_output : output) : nullptr, canonical_routes,
                    experts, width, intermediate, rows * top_k, top_k, 0,
                    uint32_t{1} << formats_.gate_up.device_execution_codebook_id,
                    uint32_t{1} << formats_.down.device_execution_codebook_id,
                    nativeVnniCodebookMaskBit(canonicalDeviceVnniCodebookId(formats_.gate_up.source_codebook_id)),
                    nativeVnniCodebookMaskBit(canonicalDeviceVnniCodebookId(formats_.down.source_codebook_id)),
                    device.ordinal, execution_stream, execution))
                throw std::runtime_error("production grouped projection launch failed: gate/up=" +
                    std::string(formats_.gate_up.label) + ", down=" + formats_.down.label +
                    ", rows=" + std::to_string(rows));
        }

        /**
         * @brief Materialize the exact same down interval consumed by BOM resolution.
         * @param ownership Domain-owned source and participant projection identity.
         * @return Persistent native descriptor bank for every domain expert.
         * @throws std::invalid_argument when the fixture is not that source layer.
         */
        DeviceNativeVNNIMatrixDesc *slice(const MoEExpertProjectionOwnership &ownership)
        {
            if (ownership.geometry() != MoEExpertProjectionOwnership::Geometry{experts, width, intermediate})
                throw std::invalid_argument("projection preparation received a different source geometry");
            const auto down = ownership.projection(WeightRole::MoEExpertDown);
            return slice(down.first_row, down.rows);
        }

        /**
         * @brief Copy exact source columns in every native packed weight plane.
         *
         * The production decoder consumes K-block-major planes. A contiguous
         * byte-offset view would be wrong: each K block needs the selected N
         * columns. Setup uses pitched D2D copies and retains original provenance.
         */
        DeviceNativeVNNIMatrixDesc *slice(int begin, int count)
        {
            const auto &metadata = *native_vnni_formats::forQuantType(formats_.down.label);
            std::vector<DeviceNativeVNNIMatrixDesc> result;
            for (const auto &source : down_host_)
            {
                auto copy = [&](const void *plane, size_t block_bytes) -> const void *
                {
                    if (!plane) return nullptr;
                    const size_t source_pitch = static_cast<size_t>(width) * block_bytes;
                    const size_t destination_pitch = static_cast<size_t>(count) * block_bytes;
                    auto *destination = allocate<uint8_t>(destination_pitch * (intermediate / 32));
                    checkProjectionHip(hipMemcpy2DAsync(destination, destination_pitch,
                        static_cast<const uint8_t *>(plane) + begin * block_bytes, source_pitch,
                        destination_pitch, intermediate / 32, hipMemcpyDeviceToDevice, stream), "copy native projection columns");
                    return destination;
                };
                auto descriptor = source;
                descriptor.n = count;
                descriptor.payload = static_cast<const uint8_t *>(copy(source.payload, metadata.payload_bytes));
                descriptor.scales = copy(source.scales, sizeof(uint16_t));
                descriptor.mins = copy(source.mins, sizeof(uint16_t));
                descriptor.emins = copy(source.emins, sizeof(uint32_t));
                result.push_back(descriptor);
            }
            return upload(result);
        }

        int width;
        int intermediate;
        int capacity;
        int experts;
        int top_k;
        DeviceId device;
        hipStream_t stream = nullptr;
        float *output = nullptr;
        int8_t *swiglu = nullptr;
        float *swiglu_scales = nullptr;

        /** @return Device-owned original-slot ownership input used by the packer. */
        const std::int32_t *routeOwners() const noexcept { return route_owners; }
        /** @return This fixture's local or complete expert-major inverse. */
        const std::int32_t *originalToGrouped() const noexcept { return inverse; }

        /** @return Original source rows for an independent quantization oracle. */
        const float *hiddenRows() const noexcept { return hidden; }
        /** @brief Replace complete input rows outside capture for numerical edge cases. */
        void publishHiddenRows(const std::vector<float> &values)
        {
            if (values.empty() || values.size() % width != 0 || values.size() > static_cast<size_t>(capacity) * width)
                throw std::invalid_argument("projection hidden publication exceeds its admitted row geometry");
            publish(hidden, values);
        }
        /** @return Published Q8 row bytes; no copy or coherence change is made. */
        const std::int8_t *quantizedHiddenRows() const noexcept { return input_q8; }
        /** @return Published per-block scales paired with quantizedHiddenRows(). */
        const float *quantizedHiddenScales() const noexcept { return input_scales; }

    private:
        /** @return One descriptor backed by a real prepared-weight lifetime owner. */
        DeviceNativeVNNIMatrixDesc prepare(const QuantizedVerifierFormatCase &format,
            int rows, int columns, unsigned seed)
        {
            weights_.push_back(format.create({static_cast<size_t>(rows), static_cast<size_t>(columns)}, seed));
            prepared_.push_back(makeGpuPreparedGemm(weights_.back().get(), device,
                "projection-boundary." + std::string(format.label) + "." + std::to_string(seed),
                ModelContextId{980000u + seed}));
            DeviceNativeVNNIMatrixDesc result;
            if (!prepared_.back().kernel->exportNativeVNNIMatrixDesc(result))
                throw std::runtime_error("projection weight descriptor export failed");
            return result;
        }

        ProjectionFixtureFormats formats_;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> buffers_;
        std::vector<std::unique_ptr<TensorBase>> weights_;
        std::vector<GpuPreparedGemm> prepared_;
        std::vector<DeviceNativeVNNIMatrixDesc> down_host_;
        DeviceNativeVNNIMatrixDesc *gate_table = nullptr, *up_table = nullptr, *down_table = nullptr;
        float *hidden = nullptr, *route_weights = nullptr;
        float *input_scales = nullptr, *gate = nullptr, *up = nullptr;
        int *counts = nullptr, *offsets = nullptr, *tokens = nullptr;
        int *inverse = nullptr, *expert_ids = nullptr, *directory = nullptr;
        int *route_owners = nullptr;
        int8_t *input_q8 = nullptr;
    };
} // namespace llaminar2::test
