/**
 * @file LocalTPCaptureSourceDefinition.h
 * @brief Optional immutable source-definition exchange in an existing cold TP boundary.
 *
 * This is not a source seal or execution authority. The existing LocalTP
 * membership/generation carries the original source cache token to its peers;
 * no second rendezvous, lookup registry or mutable GPU state is introduced.
 */
#pragma once
#include "backends/DeviceId.h"
#include <memory>
namespace llaminar2 {
class RetainedGraphParticipantPreparationStream;
/** @brief Same original graph-declared transport domain at every actual compute participant. */
struct RetainedGraphParticipantDefinitionDomain final {
    DeviceId source=DeviceId::invalid();
    const void* authority=nullptr; ///< Existing retained transport identity, never a new authority.
    bool operator==(const RetainedGraphParticipantDefinitionDomain&)const=default;
};
/** @brief One optional payload/result carried by an existing matched capture-end generation. */
struct LocalTPCaptureSourceDefinition final {
    RetainedGraphParticipantDefinitionDomain domain;
    std::shared_ptr<const RetainedGraphParticipantPreparationStream> publication; ///< Only original source may supply.
    std::shared_ptr<const RetainedGraphParticipantPreparationStream> result; ///< Exact same token on successful departure.
};
} // namespace llaminar2
