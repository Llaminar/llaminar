/**
 * @file Q8ResidualGroupedDiagnostic.h
 * @brief Explicit HIP original single-group references over production two-term operands.
 *
 * These numerical references borrow the complete existing prepared source,
 * live-row authority, scratch and publication contract. They introduce no
 * inference policy or storage owner and never select production dispatch.
 * A 32x64 tile uses 256 threads and the original one-K32-group LDS lifecycle,
 * preserving each independent ordered fold and the final two-term sum.
 * Keeping references separate preserves the broad prepared GEMM/CPU/CUDA
 * contract's compilation boundary during production qualification.
 */
#pragma once
#include "Q8ResidualCorrection.h"

namespace llaminar2 {
/**
 * @brief Diagnostic-only HIP original single-group staging with the complete ordinary two-term publication.
 * @param args Original codebook19 source, physical rows>=32, N>=64, live-row authority and admitted banks.
 * @param stream Exact non-default native stream owning publication and projection.
 * @throws std::invalid_argument Before publication for unsupported geometry/codebook or any original binding/stream violation.
 * @throws std::runtime_error On a native publication or projection submission error.
 * @note This numerical reference never changes production dispatch. It uses
 * the same banks, quantizer, independent K32 folds and final sum as production.
 * Inactive rows remain untouched. No allocation, count readback or new policy
 * is owned here; arbitrary total K retains one group per LDS retirement.
 */
void rocmQ8ResidualGroupedSingleGroupReference(const GPUQ8ResidualCorrectionArgs &args,void *stream);
/**
 * @brief Diagnostic-only HIP original single-group staging after complete original HC activation/publication.
 * @param args Original codebook19 projection, physical rows>=32, N>=64 and retained activated FP32 destination.
 * @param stream Exact non-default producer/consumer stream.
 * @throws std::invalid_argument Before publication for unsupported geometry/codebook or any original binding/stream/alias violation.
 * @throws std::runtime_error On a native publication or projection submission error.
 * @note Retains exact HC /4 SiLU, primary/residual publication and activated
 * output semantics; this is not a user-selectable precision or inference path.
 * The reference retains original single-group staging on every native target.
 */
void rocmHCResidualDownGroupedSingleGroupReference(const GPUHCResidualDownPublication &args,void *stream);
} // namespace llaminar2
