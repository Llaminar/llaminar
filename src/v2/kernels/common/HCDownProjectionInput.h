/**
 * @file HCDownProjectionInput.h
 * @brief Explicit original HC-down operands with optional sealed SingleQ8 input publication.
 *
 * The plain source selects its original quantizer. A declared publication is
 * required consumption of the norm's exact existing bank, never pointer caching.
 * Reducer-output publication composes independently through the typed method.
 */
#pragma once
namespace llaminar2 {
class TensorBase;
class HCBranchNormSingleQ8Publication;
class HCDownOutputPublication;
/** @brief Borrowed original normalized FP32 and the selected original input program. */
struct HCDownProjectionInput {
 const TensorBase& normalized; ///< Existing FP32 source, retained for injection/read.
 const HCBranchNormSingleQ8Publication* publication=nullptr; ///< Null retains original quantization.
};
}
