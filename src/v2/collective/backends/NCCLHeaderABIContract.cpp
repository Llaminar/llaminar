/**
 * @file NCCLHeaderABIContract.cpp
 * @brief Compiler proof of the isolated NCCL loader reduction ABI against its configured native header.
 *
 * The vendor's global declarations stay in this one translation unit. They
 * never enter public engine headers or the other vendor's runtime namespace.
 * Compiler builds require the exact configured header; installed tests call
 * the already-compiled proof without requiring an SDK or initializing a GPU.
 */
#include LLAMINAR_NCCL_ABI_HEADER
#include "collective/backends/NCCLDynamicLoader.h"
namespace llaminar2::nccl_dynamic {
static_assert(sizeof(ncclRedOp_t) == sizeof(::ncclRedOp_t), "NCCL reduction enum ABI size changed");
static_assert(alignof(ncclRedOp_t) == alignof(::ncclRedOp_t), "NCCL reduction enum ABI alignment changed");
static_assert(int(ncclSum) == int(::ncclSum), "NCCL SUM ABI mismatch");
static_assert(int(ncclProd) == int(::ncclProd), "NCCL PROD ABI mismatch");
static_assert(int(ncclMax) == int(::ncclMax), "NCCL MAX ABI mismatch");
static_assert(int(ncclMin) == int(::ncclMin), "NCCL MIN ABI mismatch");
static_assert(int(ncclAvg) == int(::ncclAvg), "NCCL AVG ABI mismatch");
/** @brief Expose the installed compiler proof without a native runtime or SDK lookup. */
bool configuredReductionABIContract() noexcept { return true; }
}
