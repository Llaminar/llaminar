/**
 * @file CPUWeightPreparationGeometry.h
 * @brief Code-owned transient geometry for final CPU resident and prepared weights.
 *
 * This immutable quote is consumed by the original BOM planner and the same
 * incremental CPU writer. It owns no allocation, balance, admission or live
 * accounting state. Original source identity and ranges remain source-owned.
 */
#pragma once
#include "tensors/TensorType.h"
#include "execution/config/RuntimeConfig.h"
#include "kernels/FloatingActivationOperandPrecision.h"
#include "loaders/NativeWeightLoadStaging.h"
#include <optional>
namespace llaminar2 {
/** @brief Existing final CPU ownership purpose, selected by original weight roles. */
enum class CPUWeightConsumer { ResidentTensor, PreparedGemm };
/**
 * @brief Quote the same native unit and simultaneous scratch used by the final CPU writer.
 * @param original_type Original on-disk tensor format, before explicit precision conversion.
 * @param matrix_rows Useful final matrix N extent, not original source allocation capacity.
 * @param matrix_columns Useful final matrix K extent, with original native block alignment.
 * @param destination_precision Existing explicit public weight precision policy.
 * @param consumer Actual final resident versus prepared-GEMM owner from the original weight plan.
 * @param operand_precision Existing sealed floating arithmetic contract.
 * @return Positive shared staging request, or nullopt only for direct writes into final resident storage.
 * @throws std::invalid_argument For unsupported/misaligned/empty source or precision policy.
 * @throws std::overflow_error For unrepresentable source/scratch geometry.
 * @note No entire-matrix-within-cap requirement exists. The reader windows this
 * useful unit population through the shared native+scratch arena resolver.
 */
std::optional<NativeWeightLoadStagingRequest> quoteCPUWeightPreparation(
    TensorType original_type,std::size_t matrix_rows,std::size_t matrix_columns,
    WeightPrecision destination_precision,CPUWeightConsumer consumer,
    FloatingActivationOperandPrecision operand_precision=FloatingActivationOperandPrecision::Native);
}
