/**
 * @file PreparedWeightIdentityContract.h
 * @brief One preparation and representation identity contract for runtime and BOM lowering.
 *
 * This setup-only value owns no physical allocation or byte ledger. Runtime
 * snapshots retain original projection control blocks; metadata may describe an
 * ordinary complete matrix only after its original source has been authenticated.
 */
#pragma once
#include "loaders/WeightPlan.h"
#include "tensors/TensorType.h"
#include <optional>
#include <vector>

namespace llaminar2
{
    /** @brief Whether this preparation actually includes routed expert jobs. */
    enum class PreparedExpertJobs { Excluded, Included };
    /** @brief Declarative classification before the canonical schema is consulted. */
    enum class PreparedGemmClassification { Rejected, Gemm, RequiresSchema };

    /** @brief Immutable source representation compared by existing prepared adoption. */
    class PreparedWeightIdentityContract final
    {
    public:
        /**
         * @brief Snapshot an actual binding without weakening its original owner.
         * @param binding Runtime binding after floating override selection.
         * @return No identity for a null source; otherwise exact current facts.
         */
        static std::optional<PreparedWeightIdentityContract> fromBinding(const WeightBinding &binding);
        /**
         * @brief Describe a proved complete ordinary matrix for metadata lowering.
         * @param rows Complete original output extent.
         * @param columns Complete original inner extent.
         * @param type Final prepared scalar/codebook type.
         * @param precision Final immutable operand arithmetic.
         * @return Exact full-source representation, with no projection owner.
         */
        static PreparedWeightIdentityContract completeOriginalMatrix(size_t rows, size_t columns,
            TensorType type, FloatingActivationOperandPrecision precision);
        /**
         * @brief Compare exactly the representation predicate used by adoption.
         * @param requested Candidate representation within the already checked store/device/source scope.
         * @return Whether shapes, final types, precision, slices and original projection owners agree.
         */
        bool compatible(const PreparedWeightIdentityContract &requested) const;
        /**
         * @brief Classify an actual runtime binding before its schema fallback.
         * @param binding Original source and declared residency/preparation intent.
         * @param device Exact preparation participant.
         * @param experts Whether routed expert jobs belong to this invocation.
         * @return Rejection, declared GEMM, or the unchanged schema-dependent decision.
         */
        static PreparedGemmClassification classify(const WeightBinding &binding, DeviceId device,
            PreparedExpertJobs experts);
        /**
         * @brief Classify a complete metadata matrix with no backend hint or projected parent.
         * @param role Original canonical source role.
         * @param device Exact participant; this does not query a backend.
         * @return Same role decision as runtime classification, prior to schema fallback.
         */
        static PreparedGemmClassification classifyOriginalMatrix(WeightRole role, DeviceId device);
    private:
        /** @brief Preserve all original dimensions, including non-matrix runtime bindings. */
        PreparedWeightIdentityContract(std::vector<size_t> shape, TensorType type,
            FloatingActivationOperandPrecision precision, WeightSliceSpec slice,
            std::optional<MoEExpertProjectionOwnership> projection,
            std::shared_ptr<TensorBase> original_projection);
        /** @return Whether no source coordinates or projection have been declared. */
        static bool emptySlice(const WeightSliceSpec &slice);
        /** @return Whether the exact existing full-source normalization applies. */
        bool fullSlice() const;
        std::vector<size_t> shape_;
        TensorType type_;
        FloatingActivationOperandPrecision precision_;
        WeightSliceSpec slice_;
        std::optional<MoEExpertProjectionOwnership> projection_;
        std::shared_ptr<TensorBase> original_projection_;
    };
}
