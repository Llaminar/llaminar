/**
 * @file StageParamsBase.h
 * @brief Concept-based validation for ComputeStage parameter structs
 * @author David Sanftenberg
 * @date January 2026
 *
 * This header provides compile-time validation that all stage Params structs
 * include the required common fields (device_id, mpi_ctx), preventing bugs
 * where a stage accidentally runs on CPU because it forgot to include device_id.
 *
 * DESIGN RATIONALE (C++23):
 * -------------------------
 * We use concepts instead of inheritance because:
 * 1. Designated initializers (.field = value) don't work with base class members
 * 2. Concepts provide compile-time validation without runtime overhead
 * 3. Each stage keeps clean, self-contained Params with full designated init support
 *
 * Previously, stages independently declared device_id/mpi_ctx, and if one forgot,
 * the stage would silently execute on CPU. Now:
 * 1. static_assert catches missing fields at compile time
 * 2. Each stage constructor calls setDevice(params.device_id) to set base class device
 * 3. IComputeStage::device() returns the authoritative device (not a "preference")
 *
 * USAGE:
 * ------
 * @code
 * class MyStage : public IComputeStage {
 * public:
 *     struct Params {
 *         // REQUIRED - validated by StageParamsRequired concept
 *         DeviceId device_id = DeviceId::cpu();
 *         const MPIContext* mpi_ctx = nullptr;
 *
 *         // Stage-specific fields
 *         ITensor* input = nullptr;
 *         ITensor* output = nullptr;
 *     };
 *     static_assert(StageParamsRequired<Params>);  // Compile-time check
 *
 *     explicit MyStage(Params params) : params_(std::move(params))
 *     {
 *         setDevice(params_.device_id);  // REQUIRED - sets base class device
 *     }
 *     // ...
 * };
 * @endcode
 */

#pragma once

#include "../../backends/DeviceId.h"
#include <concepts>

namespace llaminar2
{

    // Forward declarations
    class MPIContext;

    /**
     * @brief Concept ensuring a Params struct has required stage fields
     *
     * All stage Params structs MUST satisfy this concept. Use static_assert
     * in your stage class to get a clear compile error if fields are missing.
     *
     * Required fields:
     * - device_id: DeviceId for execution target
     * - mpi_ctx: const MPIContext* for distributed execution
     */
    template <typename T>
    concept StageParamsRequired = requires(T t) {
        { t.device_id } -> std::convertible_to<DeviceId>;
        { t.mpi_ctx } -> std::convertible_to<const MPIContext *>;
    };

    /**
     * @brief Macro to declare standard stage params fields
     *
     * Use this macro at the START of your Params struct to ensure consistent
     * field names, types, and defaults. This is the preferred way to add the
     * required fields.
     *
     * @code
     * struct Params {
     *     STAGE_PARAMS_COMMON_FIELDS;
     *     ITensor* my_input = nullptr;
     *     // ...
     * };
     * @endcode
     */
#define STAGE_PARAMS_COMMON_FIELDS                                      \
    /** @brief Target device for execution (default: CPU for safety) */ \
    DeviceId device_id = DeviceId::cpu();                               \
    /** @brief MPI context for distributed execution (optional) */      \
    const MPIContext *mpi_ctx = nullptr

} // namespace llaminar2
