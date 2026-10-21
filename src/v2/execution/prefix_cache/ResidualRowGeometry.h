/**
 * @file ResidualRowGeometry.h
 * @brief Exact residual-trunk rows at MTP publication and checkpoint boundaries.
 *
 * The model's declared residual width can differ from its attention/head width.
 * Immutable tensor shapes authenticate the stride before any copy or captured
 * selector is constructed; a narrower row is never a valid partial snapshot.
 * Live byte ranges and shifted-prefill scratch selection use the same width
 * contract, keeping ordinary compute scratch distinct from a widened residual.
 */
#pragma once
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
namespace llaminar2
{
/** @brief One declared FP32 residual row, independent of projection/head width. */
class ResidualRowGeometry
{
    std::size_t columns_;
public:
    /** @brief Exact scratch-role policy selected from declared row widths. */
    enum class ScratchSource { ComputeRows, ResidualRows };
    /** @brief Seal the model policy width before binding a tensor or graph. */
    explicit ResidualRowGeometry(int declaredColumns):columns_(declaredColumns)
    {
        if(declaredColumns<=0 || columns_>std::numeric_limits<std::size_t>::max()/sizeof(float))
            throw std::invalid_argument("residual row requires a positive representable declared width");
    }
    /** @return Exact live FP32 bytes in a complete residual row. */
    std::size_t rowBytes() const noexcept { return columns_*sizeof(float); }
    /** @return Declared residual columns, suitable for an authenticated selector's int ABI. */
    int columns() const noexcept { return static_cast<int>(columns_); }
    /** @brief Reject tensor layouts that would narrow or reinterpret the declared trunk. */
    void validate(std::span<const std::size_t> shape) const
    {
        if(shape.size()!=2 || shape.back()!=columns_)
            throw std::invalid_argument("residual tensor row width differs from the declared model trunk");
    }
    /**
     * @brief Resolve complete live FP32 rows within an admitted immutable matrix.
     * @param shape Producer or consumer matrix shape; its width must match this contract.
     * @param rows Actual rows consumed, independent of retained allocation capacity.
     * @return Checked live bytes, including zero for an empty payload.
     * @throws std::invalid_argument For a narrowed matrix or an out-of-capacity row range.
     * @throws std::overflow_error If the complete live payload is not addressable.
     */
    std::size_t bytesForRows(std::span<const std::size_t> shape,std::size_t rows) const
    {
        validate(shape);
        if(rows>shape.front())
            throw std::invalid_argument("residual row range exceeds its admitted matrix capacity");
        if(rows>std::numeric_limits<std::size_t>::max()/rowBytes())
            throw std::overflow_error("residual live row payload exceeds addressable bytes");
        return rows*rowBytes();
    }
    /**
     * @brief Choose the already-admitted scratch whose semantic width retains this residual.
     * @param computeColumns Ordinary attention/embedding width; never inferred from model names.
     * @param computeShape Existing compute-width normalized scratch.
     * @param residualShape Existing dedicated wide normalized-hidden scratch, if selected.
     * @param rows Bound selector rows, not a downloaded execution count.
     * @return ComputeRows for equal widths, otherwise ResidualRows after validating its full extent.
     * @throws std::invalid_argument For absent policy or an incompatible selected scratch matrix.
     */
    ScratchSource scratchSource(int computeColumns,std::span<const std::size_t> computeShape,
        std::span<const std::size_t> residualShape,std::size_t rows) const
    {
        if(computeColumns<=0)
            throw std::invalid_argument("residual scratch selection requires a positive compute width");
        const bool wide=columns_!=static_cast<std::size_t>(computeColumns);
        (void)bytesForRows(wide?residualShape:computeShape,rows);
        return wide?ScratchSource::ResidualRows:ScratchSource::ComputeRows;
    }
    /** @brief Resolve one immutable row without reading execution-owned state. */
    std::size_t elementOffset(std::span<const std::size_t> shape,std::size_t row) const
    {
        validate(shape);
        if(row>=shape.front() || row>std::numeric_limits<std::size_t>::max()/columns_)
            throw std::invalid_argument("residual row lies outside its admitted tensor geometry");
        return row*columns_;
    }
};
}
