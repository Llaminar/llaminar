/**
 * @file PlanningModelSample.h
 * @brief Exact, bounded GGUF matrix selections for startup kernel observations.
 *
 * A selection names source rows, source columns, or one expert, never an
 * arbitrary proxy shape. PlanningModelSource validates it against its retained
 * directory and loads it through ModelLoader. The returned sample retains its
 * final sample allocation and PMA lease together. A move-only receiver owns
 * unpublished final bytes; only completed I/O can seal a tensor. Expert
 * selections populate a 2-D final matrix directly, without loading their parent.
 * This is native source materialization, not model-specific
 * representation conversion, GPU preparation, or performance extrapolation.
 */
#pragma once

#include <cstddef>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include "backends/DeviceId.h"
#include "tensors/TensorType.h"
#include "memory/CPUStorageRoot.h"

namespace llaminar2
{
    class TensorBase;
    class PlanningModelSource;
    class PlanningModelSampleReceiver;
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
     * source_bytes is the logical source/wire extent. physicalBytes() prices
     * the final ModelSourcePayload root until its last tensor borrower retires.
     * The common native window reader fills that final root directly and
     * allocates no second payload under WeightLoadStaging.
     * Prepared weights, conversion scratch and execution workspace must
     * contribute their own typed demands; this object never admits capacity.
     */
    struct PlanningModelSampleGeometry
    {
        size_t n;
        size_t k;
        size_t source_bytes;
        /** @return Allocator-rounded final root bytes; source_bytes remains the wire extent. */
        [[nodiscard]] size_t physicalBytes() const
        {
            return CPUStoragePlacement::local().physicalBytes<uint8_t>(source_bytes);
        }
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
     * @brief Coupled lifetime of a final native sample matrix and its PMA claim.
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
        friend class PlanningModelSampleReceiver;
        struct Storage;
        /** @brief Seal a complete, destruction-ordered source-loading transaction. */
        explicit PlanningLoadedModelSample(std::shared_ptr<Storage> storage);
        std::shared_ptr<Storage> storage_;
    };

    /**
     * @brief Exclusive unpublished final allocation, distinct from a loaded tensor.
     *
     * Only source I/O and collective publication can borrow writable bytes. They
     * finish after their complete read/receive and inside failure consensus where
     * applicable. Sealing consumes the receiver even when metadata admission
     * fails; no partially received tensor or reusable failed writer escapes.
     */
    class PlanningModelSampleReceiver final
    {
    public:
        /** @brief Retire an unfinished allocation through its original PMA owner. */
        ~PlanningModelSampleReceiver();
        /** @brief Transfer the sole writer; the moved-from value cannot publish. */
        PlanningModelSampleReceiver(PlanningModelSampleReceiver &&) noexcept;
        /** @brief Retire this unfinished writer before transferring the new one. */
        PlanningModelSampleReceiver &operator=(PlanningModelSampleReceiver &&) noexcept;
    private:
        friend class PlanningModelSource;
        friend class PlanningExpertSamplePublication;
        friend class PlanningMatrixSamplePublication;
        /**
         * @brief Allocate the final root on the affined source/publication thread.
         * @param type Original source format, never a conversion request.
         * @param geometry Validated logical matrix and exact receive extent.
         * @param memory Sole physical admission/allocation authority.
         * @param host_device Exact CPU allocator retained through final retirement.
         * @return One exclusive final allocation, with no published tensor yet.
         * @throws std::invalid_argument For invalid geometry/allocator; allocation failures propagate.
         */
        static PlanningModelSampleReceiver allocate(TensorType type, PlanningModelSampleGeometry geometry,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device);
        /** @return Exact mutable extent, borrowed only until completion or retirement.
         * @throws std::logic_error After move, sealing or failed admission.
         */
        std::span<std::byte> receiveBytes();
        /** @return Immutable native tensor after successful admission; failure is absorbing.
         * @throws std::logic_error After retirement; source admission errors propagate without publication.
         */
        PlanningLoadedModelSample finish();
        struct Storage;
        /** @brief Adopt the sole final allocation without publishing its uninitialized bytes.
         * @param storage Exclusive source metadata and admitted final root.
         */
        explicit PlanningModelSampleReceiver(std::unique_ptr<Storage> storage);
        std::unique_ptr<Storage> storage_;
    };
}
