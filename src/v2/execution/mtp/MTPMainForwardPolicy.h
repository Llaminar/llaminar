/**
 * @file MTPMainForwardPolicy.h
 * @brief Immutable main-model topology shared by MTP setup and pipeline replay.
 *
 * A condition declares commit ownership; a grouped verifier declares its
 * outcome topology. These alternatives never confer sampler authority on a
 * pipeline follower. Mutable token/position/length rows stay device-owned.
 */
#pragma once
#include "MTPConditionForwardPurpose.h"
#include "MTPVerifierOutcomeGraph.h"
#include <variant>

namespace llaminar2
{
/** @brief Exactly one condition or grouped-verifier main-model invocation. */
using MTPMainForwardPolicy = std::variant<MTPConditionForwardPurpose, MTPVerifierOutcomeGraphMode>;
}
