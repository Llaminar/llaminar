/**
 * @file NativeRowsABIFixture.h
 * @brief Test-only passive records for the actual native-loader call boundary.
 *
 * Each backend test has its own explicitly loaded DSO and executable. Native
 * send/receive/communicator/stream addresses are observed, never dereferenced;
 * this fixture creates no GPU, performs no collective, and publishes no payload
 * receipt. The production loader remains the sole dispatch implementation.
 */
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(LLAMINAR_NATIVE_ROWS_ABI_NCCL)
#include "collective/backends/NCCLDynamicLoader.h"
namespace native_rows_abi = llaminar2::nccl_dynamic;
#elif defined(LLAMINAR_NATIVE_ROWS_ABI_RCCL)
#include "collective/backends/RCCLDynamicLoader.h"
namespace native_rows_abi = llaminar2::rccl_dynamic;
#else
#error "Select exactly one native loader ABI for this test-only fixture"
#endif
#if defined(LLAMINAR_NATIVE_ROWS_ABI_NCCL) && defined(LLAMINAR_NATIVE_ROWS_ABI_RCCL)
#error "One fixture cannot own two native loader namespaces"
#endif

/** @brief A copied host ABI observation; never native state or byte accounting. */
struct NativeRowsABICall final
{
    unsigned fixed_calls = 0;
    unsigned counted_calls = 0;
    unsigned unexpected_calls = 0;
    unsigned byte_prefix_calls = 0;
    unsigned byte_slice_calls = 0;
    unsigned authority_calls = 0;
    unsigned abort_signal_calls = 0;
    const ncclLlaminarCompletionOperand* completion = nullptr;
    std::uint64_t* live_bytes = nullptr;
    const llaminar2::NativeByteSourceSlice* source_slice = nullptr;
    std::size_t capacity_bytes = 0;
    std::size_t source_capacity_bytes = 0;
    std::uint64_t source_offset_origin = 0;
    int root = -1;
    const void *send = nullptr;
    void *receive = nullptr;
    std::size_t elements = 0;
    int datatype = -1;
    int reduction = -1; ///< Exact native reduction opcode observed at the loaded ABI boundary.
    void *communicator = nullptr;
    void *stream = nullptr;
    const std::int32_t *live_rows = nullptr;
    std::size_t capacity_rows = 0;
    std::size_t row_bytes = 0;
    unsigned long long *payload_receipt = nullptr;
};

/** @brief Resolve this exact linked DSO's installed path with dladdr. */
extern "C" void llaminarNativeRowsABIAnchor();
/** @brief Reset the isolated fixture before an actual wrapper call. */
extern "C" void llaminarNativeRowsABIReset();
/** @return Copy of the actual ABI arguments observed by this fixture. */
extern "C" NativeRowsABICall llaminarNativeRowsABIObservation();
