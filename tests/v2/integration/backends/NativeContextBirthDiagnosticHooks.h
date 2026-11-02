/** @file NativeContextBirthDiagnosticHooks.h
 * @brief Test-only creator observation/fault boundary, never a production admission API.
 */
#pragma once
#include <cstddef>
#include <functional>
#include <string>
#include <vector>
namespace llaminar2::test::birth_diagnostic {
/** @brief The exact original constructor's native resource creation roles. */
enum class Creator { PrimaryStream, PrimaryBLAS, SecondaryBLAS, OrderingEvent };
/** @brief One actual project creator observation, including its original caller DSO. */
struct Observation { Creator creator;std::size_t index;std::string caller;bool injected;bool success;void* handle;bool destroyed=false;bool destroy_success=false;int native_status=0;int destroy_status=0; };
/** @brief Bind one raw diagnostic constructor interval and optional exact failure prefix.
 * Callback runs only in this explicit diagnostic, never in captured inference.
 */
void begin(std::size_t failCreator,std::function<void(const Observation&,bool before)>);
/** @brief End the diagnostic creator interval after original construction/rollback. */
std::vector<Observation> finish();
/** @brief Read immutable creator observations without closing original destructor observation. */
std::vector<Observation> snapshot();
/** @brief Observe actual original destructor success; duplicate/missing owners fail. */
void released(Creator,void* handle,bool success,void* returnAddress,int nativeStatus);
/** @brief Check exact caller ownership; vendor-internal allocations are not project creators. */
bool isProjectCaller(void* returnAddress);
/** @brief Before an actual creator, return false only for the selected injected diagnostic failure. */
bool before(Creator,void* returnAddress);
/** @brief Preserve exact original success/failure/handle observation without owning the handle. */
void after(Creator,bool success,void* handle,void* returnAddress,int nativeStatus);
} // namespace llaminar2::test::birth_diagnostic
