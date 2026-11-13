/**
 * @file RCCLHeaderABIContract.cpp
 * @brief Compiler proof of the isolated RCCL loader reduction ABI against its configured native header.
 *
 * The vendor's global declarations stay in this one translation unit. They
 * never enter public engine headers or the other vendor's runtime namespace.
 * Compiler builds require the exact configured header; installed tests call
 * the already-compiled proof without requiring an SDK or initializing a GPU.
 */
#include LLAMINAR_RCCL_ABI_HEADER
#include "collective/backends/RCCLDynamicLoader.h"
namespace llaminar2::rccl_dynamic {
static_assert(sizeof(ncclRedOp_t) == sizeof(::ncclRedOp_t), "RCCL reduction enum ABI size changed");
static_assert(alignof(ncclRedOp_t) == alignof(::ncclRedOp_t), "RCCL reduction enum ABI alignment changed");
static_assert(int(ncclSum) == int(::ncclSum), "RCCL SUM ABI mismatch");
static_assert(int(ncclProd) == int(::ncclProd), "RCCL PROD ABI mismatch");
static_assert(int(ncclMax) == int(::ncclMax), "RCCL MAX ABI mismatch");
static_assert(int(ncclMin) == int(::ncclMin), "RCCL MIN ABI mismatch");
static_assert(int(ncclAvg) == int(::ncclAvg), "RCCL AVG ABI mismatch");
/** @brief Expose the installed compiler proof without a native runtime or SDK lookup. */
bool configuredReductionABIContract() noexcept { return true; }
}
