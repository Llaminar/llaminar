/**
 * @file ROCmDiagnosticSnapshotCopy.hip
 * @brief HIP compilation boundary for exact authored diagnostic snapshot copies.
 *
 * The shared integer scan captures row counts on the producer's native stream,
 * and parallel bit copies retain only live values in private diagnostic slots.
 */
#define DIAGNOSTIC_SNAPSHOT_ROCM 1
#include "../../common/GPUDiagnosticSnapshotCopy.cuh"
