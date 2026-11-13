/**
 * @file CUDADiagnosticSnapshotCopy.cu
 * @brief CUDA compilation boundary for exact authored diagnostic snapshot copies.
 *
 * The generic diagnostic authority supplies stable arena slots and native
 * count bindings; these kernels introduce no model execution state or padding.
 */
#define DIAGNOSTIC_SNAPSHOT_CUDA 1
#include "../../common/GPUDiagnosticSnapshotCopy.cuh"
