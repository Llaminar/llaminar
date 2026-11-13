/**
 * @file CPUPreparedWeightWriter.h
 * @brief Incremental final CPU weights with original native source coordinates.
 *
 * The original loader/FrozenWeightPlan retain source and selection authority.
 * This consumer owns one admitted final root and no source/converted matrix
 * shadow. Original source windows are borrowed only during append. Nothing is
 * publishable until every exact source coordinate has committed successfully.
 */
#pragma once
#include "tensors/NativeWeightSource.h"
#include "tensors/NativeWeightSourceWindow.h"
#include "CPUWeightPreparationGeometry.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "memory/CPUWeightStoragePlacement.h"
#include <functional>
#include <memory>
#include <span>
namespace llaminar2 {
class TensorBase;
class ITensorGemm;
/** @brief Original conversion contract selected by the existing model representation owner. */
enum class CPUPreparedHalfConversion { GenericTensor, ModelPreparedHalf };
/** @brief Same original matrix/range and resolved precision supplied by the original producer. */
struct CPUPreparedWeightSource {
    NativeWeightSourceGeometry original;
    NativeWeightSourceWindow range;
    size_t matrix_count=1; ///< Complete original outer population for resident storage; GEMM requires exactly one.
    WeightPrecision destination_precision=WeightPrecision::NATIVE;
    CPUWeightConsumer consumer=CPUWeightConsumer::PreparedGemm;
    FloatingActivationOperandPrecision operand_precision=FloatingActivationOperandPrecision::Native;
    CPUProjectionNumericalPolicy numerical_policy=CPUProjectionNumericalPolicy::BackendNative;
    CPUPreparedHalfConversion half_conversion=CPUPreparedHalfConversion::GenericTensor; ///< Model overrides retain canonical half bits/validation; generic conversion retains its Tensor arithmetic.
    std::optional<NativeWeightSourceAssembly> assembly; ///< Exact original stripes, with cumulative destinations validated before final allocation.
};
/** @brief Committed actual final roots; native-packed metadata is created by Store after this result. */
struct CPUPreparedWeightResult {
    std::shared_ptr<ITensorGemm> kernel; ///< Actual prepared engine for PreparedGemm, absent for resident-only consumers.
    std::shared_ptr<TensorBase> final_tensor; ///< Genuine final resident tensor when aliased by execution; absent for native packed storage.
};
/** @brief One final admitted allocation, exact ordered source coverage and absorbing failure. */
class CPUPreparedWeightWriter final {
public:
    /** @brief Validate original geometry/policy and allocate final storage once under the supplied original PMA. */
    static std::unique_ptr<CPUPreparedWeightWriter> begin(CPUPreparedWeightSource source,CPUWeightStoragePlacement placement);
    ~CPUPreparedWeightWriter();
    CPUPreparedWeightWriter(const CPUPreparedWeightWriter&)=delete;
    CPUPreparedWeightWriter& operator=(const CPUPreparedWeightWriter&)=delete;
    /** @return Same immutable actual source/range/policy; no source readiness is inferred. */
    const CPUPreparedWeightSource& source() const & noexcept;
    /** @return Same code-owned source/scratch quote used by the original planner; nullopt means direct final writes. */
    std::optional<NativeWeightLoadStagingRequest> stagingRequest() const;
    /**
     * @brief Consume one exact native rectangle into final storage, with all concurrent scratch in the admitted arena.
     * @param window Actual original coordinates; whole useful rows or one row of native-group K strips.
     * @param native Exact useful native bytes, borrowed synchronously and never retained.
     * @param scratch Consumer subrange of the SAME admitted source arena.
     * @throws std::invalid_argument For foreign, duplicate, gapped or unaligned input or insufficient scratch.
     */
    void append(const NativeWeightSourceWindow& window,std::span<const std::byte> native,std::span<std::byte> scratch);
    /**
     * @brief Read an unchanged representation directly into its one final root without transient source storage.
     * @param window Next exact original rectangle in source order.
     * @param read Original loader callback, given only the genuine useful final destination span.
     * @throws std::exception For invalid geometry/policy or read failure; failure prevents publication.
     */
    void readDirect(const NativeWeightSourceWindow& window,const std::function<void(std::span<std::byte>)>& read);
    /** @brief Publish actual final ownership only after complete coverage; missing/failed/duplicate completion rejects. */
    CPUPreparedWeightResult finish();
private:
    struct Impl;
    explicit CPUPreparedWeightWriter(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}
