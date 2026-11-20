/**
 * @file NativeMPISessionSchizoBridge.h
 * @brief C ABI boundary between genuine OpenMPI internal headers and owned C++ policy.
 *
 * OpenMPI 4.1.6's AArch64 internal atomic headers are C interfaces. Only the C
 * adapter names ORTE structures/component tables; C++ retains the single typed
 * environment/readiness authority. No SDK macros, atomic emulation or duplicate
 * environment schema is needed at this process-startup boundary.
 */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
/** @brief Apply owned policy to the original daemon/application environment.
 * @param daemon_environment Borrowed original ORTE daemon environment.
 * @param application_environment Native application environment publication slot,
 * or null when the native application context is missing.
 * @return Genuine ORTE success/error code; unowned jobs remain unchanged.
 */
int llaminar_mpi_native_setup_fork(char **daemon_environment, char ***application_environment);
/** @brief Admit the native daemon component using live typed protocol/ABI state.
 * @return Genuine ORTE success for an admitted owned daemon, or its error code.
 */
int llaminar_mpi_native_query_protocol(void);
#ifdef __cplusplus
}
#endif
