/**
 * @file PlanningModelMetadata.h
 * @brief One metadata-only model descriptor for cluster planning and admission.
 *
 * A descriptor retains the exact GGUF tensor inventory and the learned-MTP
 * aware main-layer boundary. Rank zero reads it once; all ranks receive the
 * same immutable bytes before deriving placement or memory BOM inputs. It
 * owns no TensorFactory, prepared weights, backend context or live byte ledger.
 */
#pragma once

#include "planning/ModelMemoryProfile.h"
#include "planning/PlanningModelSample.h"
#include <functional>
#include <memory>
#include <span>

namespace llaminar2
{
    class ModelLoader;
    class IMPIContext;
    class PhysicalMemoryAuthority;
    class TensorBase;
    struct DeviceId;
    struct ModelConfig;
    /** @brief Validated, serializable planning geometry with no execution state. */
    class PlanningModelMetadata final
    {
    public:
        /**
         * @brief Bind the physical tensor directory to its main forward-layer interval.
         * @param profile Exact GGUF metadata and native tensor inventory.
         * @param main_layer_count Layer count resolved by the learned MTP manifest.
         * @throws std::invalid_argument for invalid main-layer or attention geometry.
         */
        PlanningModelMetadata(ModelMemoryProfile profile, int main_layer_count);

        /** @return Descriptor projected from one already parsed authoritative GGUF loader. */
        [[nodiscard]] static PlanningModelMetadata fromLoader(const ModelLoader &loader);

        /** @return Exact immutable metadata consumed by canonical BOM construction. */
        [[nodiscard]] const ModelMemoryProfile &memoryProfile() const noexcept { return profile_; }
        /** @return Main forward-layer count, excluding appended learned MTP layers. */
        [[nodiscard]] int mainLayerCount() const noexcept { return main_layer_count_; }
        /** @return ExecutionPlanBuilder's view, derived rather than maintained separately. */
        [[nodiscard]] ModelConfig executionModelConfig() const;
        /** @return Versioned wire bytes containing the existing profile format unchanged. */
        [[nodiscard]] std::vector<uint8_t> serialize() const;
        /** @return Validated descriptor; truncated, stale or malformed payloads throw. */
        [[nodiscard]] static PlanningModelMetadata deserialize(std::span<const uint8_t> bytes);

    private:
        ModelMemoryProfile profile_;
        int main_layer_count_;
    };

    /**
     * @brief Scoped metadata-only GGUF owner for a complete automatic search.
     *
     * The loader and its process-local factory live together at stable addresses.
     * A const loader view exposes the existing tensor/MTP directory to BOM
     * builders without copying another inventory into a wire format. Construction
     * maps or loads no payload. Explicit, admitted matrix samples may subsequently
     * read bounded source slices; they never map the full model or prepare kernels.
     * Only rank zero constructs this owner during distributed discovery; peers
     * receive the ordinary compact descriptor.
     */
    class PlanningModelSource final
    {
    public:
        /** @brief Parse one real GGUF (including split metadata), failing on invalid input. */
        explicit PlanningModelSource(const std::string &model_path);
        /** @brief Retire the metadata loader before its factory and MPI context. */
        ~PlanningModelSource();
        /** @brief Move stable ownership without relocating a loader's borrowed factory. */
        PlanningModelSource(PlanningModelSource &&) noexcept;
        /** @brief Replace the owned metadata transaction without moving internal addresses. */
        PlanningModelSource &operator=(PlanningModelSource &&) noexcept;
        /** @return Immutable descriptor; a moved-from source is a lifecycle error. */
        const PlanningModelMetadata &metadata() const;
        /** @return Const metadata loader; payload-loading APIs remain unavailable. */
        const ModelLoader &loader() const;
        /** @return Exact source-path intent used to authenticate a candidate's model. */
        const std::string &path() const;

        /**
         * @brief Resolve one source matrix without reading any weight bytes.
         * @param request Whole matrix, row/column slice, or one expert.
         * @return Exact N/K and native source extent for PMA BOM contribution.
         * @throws std::invalid_argument for unsupported dimensions, formats or ranges.
         * @throws std::overflow_error when any source extent cannot be represented.
         */
        PlanningModelSampleGeometry sampleGeometry(const PlanningModelSampleRequest &request) const;

        /**
         * @brief Load one admitted native matrix through the retained production loader.
         * @param request Same selection used to build the sample BOM.
         * @param memory Rank-bound authority admitting source and transient read bytes.
         * @param host_device Exact CPU allocator; GPU preparation is a separate operation.
         * @return A 2-D sample owner retaining source storage and its allocation lease.
         * @throws std::invalid_argument for absent authority, non-CPU placement or bad geometry.
         * @throws std::runtime_error if the bounded source read or view construction fails.
         *
         * Claims precede all payload allocation. The temporary reader claim retires
         * on return; the source claim survives this object with the loaded sample.
         * No tensor/format conversion, hidden payload cache or full-model mapping occurs.
         */
        PlanningLoadedModelSample loadSample(const PlanningModelSampleRequest &request,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device) const;

    private:
        struct Storage;
        std::unique_ptr<Storage> storage_;
    };

    /**
     * @brief Read metadata and tensor extents without mapping/loading weight payloads.
     * @param model_path GGUF path, including its ordinary split-file convention.
     * @return Complete descriptor; unreadable or invalid GGUF is a hard error.
     */
    PlanningModelMetadata readPlanningModelMetadata(const std::string &model_path);

    struct GGUFModel;
    /**
     * @brief Bound initial upload slots from the exact parsed tensor directory.
     * @param model Complete GGUF metadata, including split-file tensors.
     * @return Largest source tensor; row slices cannot exceed their parent.
     * @throws std::invalid_argument for an empty payload directory.
     * @throws std::overflow_error when the extent exceeds addressable memory.
     */
    size_t maximumGGUFTensorPayloadBytes(const GGUFModel &model);

    /**
     * @brief Publish a root-owned descriptor with rank-synchronous failure handling.
     * @param mpi Exact admission communicator; null denotes a process-local request.
     * @param root_reader Called only on communicator rank zero, once per invocation.
     * @return Identical descriptor on every rank after complete validation consensus.
     * @throws std::runtime_error if root reading, allocation, publication or validation fails.
     *
     * This startup-only exchange uses the existing RankInitializationLifecycle
     * for every fallible local phase. A throwing reader is consensus data, never
     * permission for one rank to leave followers waiting in a later broadcast.
     */
    PlanningModelMetadata exchangePlanningModelMetadata(
        const std::shared_ptr<IMPIContext> &mpi,
        const std::function<PlanningModelMetadata()> &root_reader);
}
