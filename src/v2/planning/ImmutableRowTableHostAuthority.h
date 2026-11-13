/**
 * @file ImmutableRowTableHostAuthority.h
 * @brief One canonical rank-local CPU storage observation, independent of NUMA execution affinity.
 * NUMA worker placement never creates a second capacity bank; the public CPU
 * backend accepts only CPU0 and installs locality through its explicit policy.
 */
#pragma once
#include "planning/PhysicalMemoryBOM.h"
#include <stdexcept>
namespace llaminar2 {
/**
 * @brief Reject invented CPU allocator identities before lookup contribution/admission.
 * @param host Original observed canonical rank-local CPU0 resource.
 * @throws std::invalid_argument For invalid resources or a nonzero physical CPU identity.
 */
inline void requireCanonicalImmutableHostObservation(PhysicalMemoryResource host){
    if(!host.valid() || host.device!=DeviceId::cpu())
        throw std::invalid_argument("Ngram physical host allocator must be canonical CPU0; NUMA affinity belongs to host placement policy");
}
}
