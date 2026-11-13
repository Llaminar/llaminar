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
#include "loaders/SharedPredictorGGUFSource.h"
#include "planning/PlanningModelSample.h"
#include "loaders/WeightIdentity.h"
#include <functional>
#include <memory>
#include <span>
#include <map>
#include <string_view>
#include <cstdint>

namespace llaminar2
{
    class ModelLoader;
    class IMPIContext;
    class PhysicalMemoryAuthority;
    class TensorBase;
    struct DeviceId;
    struct ModelConfig;
    class PlanningModelMetadata;
    class PlanningTensorIndex;
    /** @brief Borrowed, preclassified original tensor; only its immutable metadata owner can mint it. */
    class PlanningClassifiedTensor final
    {
    public:
        /** @return Exact original source entry; the metadata owner must outlive this view. */
        const TensorSizeInfo &tensor() const noexcept { return owner_->tensors[index_]; }
        /** @return Canonical role retained once, independently of placement or arithmetic policy. */
        WeightRole role() const noexcept { return role_; }
        /** @return Original source index, not a participant-local tensor ordinal. */
        size_t index() const noexcept { return index_; }
        /** @return Exact immutable source authority, allowing resolvers to reject foreign views. */
        const ModelMemoryProfile &profile() const noexcept { return *owner_; }
    private:
        friend class PlanningModelMetadata;
        friend class PlanningTensorIndex;
        PlanningClassifiedTensor(const ModelMemoryProfile &owner,size_t index,WeightRole role)
            :owner_(&owner),index_(index),role_(role) {}
        const ModelMemoryProfile *owner_;
        size_t index_;
        WeightRole role_;
    };
    /** @brief One immutable canonical profile and its derived source-only role/layer/name indices. */
    class PlanningTensorIndex final
    {
    public:
        /** @brief Move the original profile once; retained indices never duplicate tensor payload metadata. */
        explicit PlanningTensorIndex(ModelMemoryProfile profile);
        PlanningTensorIndex(const PlanningTensorIndex &)=delete;
        PlanningTensorIndex &operator=(const PlanningTensorIndex &)=delete;
        /** @return Original immutable source profile. */
        const ModelMemoryProfile &profile() const noexcept { return profile_; }
        /** @return Source-bound role without parsing a name during candidate search. */
        PlanningClassifiedTensor classifiedTensor(size_t index) const;
        /** @return Original non-layer tensor indices. */
        std::span<const size_t> globalTensorIndices() const noexcept { return globals_; }
        /** @return Exact original indices for one layer. */
        std::span<const size_t> layerTensorIndices(int layer) const;
        /** @brief Visit every original source tensor in an explicit interval, including appended sidecar layers. */
        template<class Visitor> void forEachLayerTensor(int first,int last,Visitor &&visit) const {
            for(auto layer=layers_.lower_bound(first);layer!=layers_.end()&&layer->first<=last;++layer)
                for(const size_t index:layer->second)visit(index);
        }
        /** @return First original tensor with the exact canonical role. */
        std::optional<size_t> firstTensorWithRole(WeightRole role) const noexcept;
        /** @return Original tensor for an exact canonical name, absent when not declared. */
        const TensorSizeInfo *findTensor(std::string_view name) const noexcept;
        /** @return Canonical role for an original element, absent for a deliberate synthetic semantic alias. */
        std::optional<WeightRole> roleForOriginal(const TensorSizeInfo &tensor) const noexcept;
        /** @return Sealed original view, absent for semantic aliases not stored in this source. */
        std::optional<PlanningClassifiedTensor> classifyOriginal(const TensorSizeInfo &tensor) const noexcept;
        /** @return First original source with exact layer and suffix, without scanning other layers. */
        const TensorSizeInfo *findLayerTensor(int layer,std::string_view suffix) const noexcept;
        /** @brief Reject accidental reuse for another profile, even an equal-shaped source copy. */
        void requireProfile(const ModelMemoryProfile &profile) const;
    private:
        const ModelMemoryProfile profile_;
        std::vector<WeightRole> roles_;
        std::vector<size_t> globals_;
        std::map<int,std::vector<size_t>> layers_;
        std::map<WeightRole,size_t> first_roles_;
        std::map<std::string_view,size_t,std::less<>> names_;
        std::map<std::pair<int,std::string_view>,size_t> layer_suffixes_;
    };

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
        [[nodiscard]] const ModelMemoryProfile &memoryProfile() const noexcept { return source_index_->profile(); }
        /** @return Main forward-layer count, excluding appended learned MTP layers. */
        [[nodiscard]] int mainLayerCount() const noexcept { return main_layer_count_; }
        /** @return Source-bound classification without parsing a tensor name during candidate search. */
        PlanningClassifiedTensor classifiedTensor(size_t index) const;
        /** @return Original non-layer tensor indices in original directory order. */
        std::span<const size_t> globalTensorIndices() const noexcept { return source_index_->globalTensorIndices(); }
        /** @return Exact original indices for one layer, with no full-directory scan. */
        std::span<const size_t> layerTensorIndices(int layer) const;
        /** @return First exact role match, retained once; absent roles remain absent. */
        std::optional<size_t> firstTensorWithRole(WeightRole role) const noexcept;
        /** @return Shared immutable index; safe to retain in pure quote inputs beyond discovery. */
        const std::shared_ptr<const PlanningTensorIndex> &tensorIndex() const noexcept { return source_index_; }
        /** @return ExecutionPlanBuilder's view, derived rather than maintained separately. */
        [[nodiscard]] ModelConfig executionModelConfig() const;
        /** @return Versioned wire bytes containing the existing profile format unchanged. */
        [[nodiscard]] std::vector<uint8_t> serialize() const;
        /** @return Validated descriptor; truncated, stale or malformed payloads throw. */
        [[nodiscard]] static PlanningModelMetadata deserialize(std::span<const uint8_t> bytes);

    private:
        std::shared_ptr<const PlanningTensorIndex> source_index_;
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
        explicit PlanningModelSource(const std::string &model_path,
            const std::optional<SharedPredictorGGUFSource> &predictor = {});
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
        /** @return Exact optional separate predictor selection authenticated with the target. */
        const std::optional<SharedPredictorGGUFSource> &predictorSource() const;

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
         * @param memory Rank-bound authority admitting the final native source sample.
         * @param host_device Exact CPU allocator; GPU preparation is a separate operation.
         * @return A 2-D sample owner retaining source storage and its allocation lease.
         * @throws std::invalid_argument for absent authority, non-CPU placement or bad geometry.
         * @throws std::runtime_error if the bounded original-source read fails.
         *
         * Claims precede all payload allocation. The common window reader writes
         * directly into the final sample; no matrix-sized reader buffer exists.
         * The source claim survives this object with the loaded sample.
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
    PlanningModelMetadata readPlanningModelMetadata(const std::string &model_path,
        const std::optional<SharedPredictorGGUFSource> &predictor = {});

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
