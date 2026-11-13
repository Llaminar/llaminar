/** @file RetainedHostServiceQualifiedLimits.h
 * @brief Tested mapped-pthread role families; exact rank geometry is carried separately.
 */
#pragma once
#include "RetainedHostServiceClassPolicy.h"
#include "RetainedHostServiceProgram.h"
#include <array>
#include <string_view>
namespace llaminar2::detail {
/** @brief Supported creator/role and retained evidence; no empirical support ceiling or historical sizeof. */
struct RetainedHostServiceQualifiedClass final {RetainedHostServiceRole role;RetainedHostServiceCreatorVersion creatorVersion;std::string_view evidence;};
inline constexpr std::array<RetainedHostServiceQualifiedClass,2> retainedHostServiceQualifiedClasses{{
 {RetainedHostServiceRole::ImmutableLookupIO,RetainedHostServiceCreatorVersion::MappedPthreadV1,"b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce"},
 {RetainedHostServiceRole::ExecutorParentTicket,RetainedHostServiceCreatorVersion::MappedPthreadV1,"b82fa882e6bb6b7a5175df363bd1f591175c8754000afbbd2c94ac943d7fd7ce"}
}};
}
