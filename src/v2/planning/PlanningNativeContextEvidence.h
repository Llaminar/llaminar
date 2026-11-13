/** @file PlanningNativeContextEvidence.h
 * @brief Immutable original-rank native family declarations, without live PMA
 * ownership. Source-owned pthread/object metadata is collected once; selected
 * membership projects source-installed contracts into the same fixed BOM before
 * quota admission.
 */
#pragma once
#include "backends/NativeContextOpaqueContract.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <map>
#include <vector>
namespace llaminar2 {
class AutomaticPlanningPreparation;
class ExecutionRankMembership;
/** @brief Original discovery-rank declarations, never caller opaque byte
 * limits. */
struct PlanningNativeContextDeclaration final {
  DeviceId device;
  std::string physical_uuid;
  std::size_t stack_bytes, guard_bytes, page_bytes, object_bytes,
      object_alignment;
};
using PlanningNativeContextBudgets =
    std::map<int, std::vector<PlanningNativeContextDeclaration>>;
/** @return Original declarations for selected execution rank; missing
 * membership evidence is fatal. */
const std::vector<PlanningNativeContextDeclaration> &
projectPlanningNativeContextBudget(const PlanningNativeContextBudgets &,
                                   const ExecutionRankMembership &, int);
/** @brief Exchange original stack/object geometry without native work or
 * provider hashing. */
PlanningNativeContextBudgets
collectPlanningNativeContextBudgets(const AutomaticPlanningPreparation &);
/** @brief Project exact selected worker opaque and stack/object obligations
 * once per physical device.
 * @param keys Original stack/object declarations. @param host Original CPU0
 * resource.
 * @param devices Selected GPU resources in execution namespace. @return Typed
 * fixed-owner contribution.
 */
PhysicalMemoryPlan
planningNativeContextBOM(std::span<const PlanningNativeContextDeclaration> keys,
                         const PhysicalMemoryResource &host,
                         std::span<const PhysicalMemoryResource> devices);
} // namespace llaminar2
