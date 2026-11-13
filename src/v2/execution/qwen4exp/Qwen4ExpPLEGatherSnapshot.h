/**
 * @file Qwen4ExpPLEGatherSnapshot.h
 * @brief Exact diagnostic view of the original decoded FP32 lexical gather output.
 *
 * The descriptor borrows the graph's existing output tensor. It never exposes
 * native transport packets or controller words, copies data, or changes tensor
 * ownership. SnapshotCapture retains its ordinary arena/event publication path.
 */
#pragma once
#include "execution/compute_stages/IComputeStage.h"
#include "tensors/TensorClasses.h"
#include <limits>
#include <stdexcept>

namespace llaminar2 {
/** @brief Immutable semantic geometry of one original lexical gather output. */
class Qwen4ExpPLEGatherSnapshot final {
public:
    /**
     * @brief Bind the existing final decoder output without materializing storage.
     * @param tensor Original FP32 arena output, retained by the graph operands.
     * @param rows Actual physical rows of this retained graph variant.
     * @param columns Complete original-head-order decoded embedding width.
     * @param offset Exact aligned byte subview in the original tensor.
     * @throws std::invalid_argument For wrong type, empty geometry, or an invalid original subview.
     */
    Qwen4ExpPLEGatherSnapshot(TensorBase &tensor, std::size_t rows, std::size_t columns, std::size_t offset = 0)
        : tensor_(&tensor), rows_(rows), columns_(columns), offset_(offset)
    {
        if (tensor.native_type() != TensorType::FP32 || !rows || !columns || offset % sizeof(float) ||
            columns > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            rows > std::numeric_limits<std::size_t>::max() / (columns * sizeof(float)) ||
            offset > tensor.size_bytes() || rows * columns * sizeof(float) > tensor.size_bytes() - offset)
            throw std::invalid_argument("PLE gather snapshot requires its exact original FP32 output subview");
    }
    /** @return Only the final FP32 gathered rows, through the original tensor/event snapshot authority. */
    StageDumpInfo describe() const
    {
        const auto *base = static_cast<const std::byte *>(tensor_->raw_data());
        StageDumpInfo dump;
        dump.outputs.push_back({"gathered_rows", base ? base + offset_ : nullptr, rows_, columns_, "FP32",
            sizeof(float), rows_ * columns_ * sizeof(float), tensor_, {}, offset_});
        return dump;
    }
private:
    TensorBase *const tensor_; ///< Borrowed original output; graph operands own its lifetime.
    const std::size_t rows_, columns_, offset_;
};
} // namespace llaminar2
