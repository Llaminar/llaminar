/**
 * @file PlanningModelSample.h
 * @brief Exact, bounded GGUF matrix selections for startup kernel observations.
 *
 * A selection names source rows, source columns, or one expert, never an
 * arbitrary proxy shape. PlanningModelSource validates it against its retained
 * directory and loads it through ModelLoader. The returned sample retains its
 * source allocation and PMA lease together, including when it is a 2-D view of
 * one 3-D expert. This is native source materialization, not model-specific
 * representation conversion, GPU preparation, or performance extrapolation.
 */
#pragma once

#include <cstddef>
#include <array>
#include <memory>
#include <string>
#include <variant>
#include "backends/DeviceId.h"
#include "tensors/TensorType.h"

namespace llaminar2
{
    class TensorBase;
    class PlanningModelSource;
    class PhysicalMemoryAuthority;
    /** @brief An entire ordinary 2-D source matrix. */
    struct PlanningWholeMatrix {};
    /** @brief Half-open N interval of an ordinary 2-D source matrix. */
    struct PlanningMatrixRows { size_t first; size_t last; };
    /** @brief Half-open, source-codebook-aligned K interval of a 2-D matrix. */
    struct PlanningMatrixColumns { size_t first; size_t last; };
    /** @brief One complete matrix from a [K,N,experts] GGUF parent. */
    struct PlanningExpertMatrix { size_t index; };

    /** @brief Mutually exclusive source selection; validated before admission or I/O. */
    struct PlanningModelSampleRequest
    {
        std::string tensor_name;
        std::variant<PlanningWholeMatrix, PlanningMatrixRows,
            PlanningMatrixColumns, PlanningExpertMatrix> selection;
    };

    /**
     * @brief Validated native matrix geometry and exact source-loading BOM input.
     *
     * source_bytes belongs to ModelSourcePayload until the last tensor borrower
     * retires. ModelLoader's non-mapped slice reader temporarily holds another
     * source_bytes under WeightLoadStaging. These are distinct real allocations,
     * not an anonymous reserve. Prepared weights and execution workspace must
     * contribute their own typed demands; this object never admits capacity.
     */
    struct PlanningModelSampleGeometry
    {
        size_t n;
        size_t k;
        size_t source_bytes;
    };

    /** @brief Complete gate/up/down source identity, shared by every backend. */
    struct PlanningExpertSampleRequest final
    {
        PlanningModelSampleRequest gate;
        PlanningModelSampleRequest up;
        PlanningModelSampleRequest down;

        /** @return Projections in FFN arithmetic order; no tensor ownership escapes. */
        std::array<const PlanningModelSampleRequest *, 3> projections() const
        {
            return {&gate, &up, &down};
        }
    };

    /** @brief Validated full expert geometry; contains no execution or live allocation state. */
    struct PlanningExpertSampleDescription
    {
        std::array<PlanningModelSampleGeometry, 3> matrices;
        std::array<std::string, 3> formats;
        int layer;
        size_t source_bytes;
        size_t largest_source_bytes;

        /**
         * @brief Validate one same-layer/source expert before I/O on any backend.
         * @throws std::invalid_argument for incompatible or partial projections.
         * @throws std::overflow_error for a source extent that cannot be represented.
         */
        static PlanningExpertSampleDescription resolve(const PlanningModelSource &source,
            const PlanningExpertSampleRequest &request);
    };

    /**
     * @brief Coupled lifetime of a loaded matrix, its parent and its PMA claim.
     *
     * Tensor views can internally retain a parent's shared_from_this() owner.
     * Returning an aliasing shared_ptr with a separate lease owner would let a
     * nested view bypass the lease. Instead this value owns the entire sample
     * and exposes only a const borrow, not an independent tensor owner.
     * Retain this value for the entire prepared-kernel/sample transaction.
     */
    class PlanningLoadedModelSample final
    {
    public:
        /** @return Const matrix valid until this sample's last owning copy retires. */
        const TensorBase &tensor() const;
    private:
        friend class PlanningModelSource;
        friend class PlanningExpertSamplePublication;
        friend class PlanningMatrixSamplePublication;
        /** @brief Allocate an unpublished native receive tensor after its physical claim. */
        static PlanningLoadedModelSample allocateReceive(TensorType type, PlanningModelSampleGeometry geometry,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device);
        /** @return Mutable bytes exclusively for the enclosing unpublished receive transaction. */
        void *receiveData();
        struct Storage;
        /** @brief Seal a complete, destruction-ordered source-loading transaction. */
        explicit PlanningLoadedModelSample(std::shared_ptr<Storage> storage);
        std::shared_ptr<Storage> storage_;
    };
}
