/**
 * @file DeviceGenerationInitialization.h
 * @brief Typed immutable request identity for resident response-window admission.
 *
 * Backends pass this descriptor unchanged to the controller initializer. Its
 * device ticket pointer authenticates continuation without exposing mutable
 * depth state to the host or allocating a second copy of that state.
 */
#pragma once
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_GENERATION_ADMISSION_HD __host__ __device__
#else
#define LLAMINAR_GENERATION_ADMISSION_HD
#endif
namespace llaminar2::sampling_math
{
    struct DeviceGenerationDispatchTicket;
    /** @brief Distinguish a new request from another publication of its response. */
    enum class DeviceGenerationAdmissionKind : int
    {
        NewRequest,
        ContinueResponse,
    };

    /**
     * @brief Immutable identity for initializing resident response-window data.
     *
     * A continuation authenticates the prior ticket on device before retaining
     * learner state. The ticket and controller already have persistent arena
     * storage; no host depth shadow or additional execution-state allocation is
     * introduced. At the backend boundary prior_tickets names all request rows;
     * each kernel lane selects its own ticket before calling the scalar helper.
     */
    struct DeviceGenerationInitialization
    {
        DeviceGenerationAdmissionKind kind = DeviceGenerationAdmissionKind::NewRequest;
        uint64_t session_epoch = 0; ///< Immutable request identity, never a live counter.
        uint64_t workspace_generation = 0; ///< Arena identity owning the retained row.
        const DeviceGenerationDispatchTicket *prior_tickets = nullptr; ///< Existing device ticket storage.

        /** @return Whether the immutable descriptor can authenticate its boundary. */
        LLAMINAR_GENERATION_ADMISSION_HD bool valid() const
        {
            return kind == DeviceGenerationAdmissionKind::NewRequest ||
                (kind == DeviceGenerationAdmissionKind::ContinueResponse &&
                 session_epoch != 0 && workspace_generation != 0 && prior_tickets);
        }

    };


}
#undef LLAMINAR_GENERATION_ADMISSION_HD
