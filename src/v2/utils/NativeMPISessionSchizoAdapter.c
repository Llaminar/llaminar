/**
 * @file NativeMPISessionSchizoAdapter.c
 * @brief Genuine C-language ORTE/MCA adapter for the owned native fork hook.
 *
 * OpenMPI 4.1.6's AArch64 internal atomic API is defined for C translation
 * units. Keep its structures and component initializers here, rather than
 * overriding SDK macros or emulating atomics in C++. Owned environment policy
 * and readiness remain in the single C++ authority behind the narrow C ABI.
 */
#include "utils/NativeMPISessionSchizoBridge.h"
#include "orte_config.h"
#include "ompi/version.h"
#include "orte/version.h"
#include "orte/constants.h"
#include "orte/runtime/orte_globals.h"
#include "orte/mca/schizo/schizo.h"
#include <stddef.h>

#if OMPI_MAJOR_VERSION != 4 || OMPI_MINOR_VERSION != 1 || OMPI_RELEASE_VERSION != 6
#error "Owned MPI native environment component requires the selected OpenMPI 4.1.6 headers"
#endif
_Static_assert(sizeof(orte_schizo_base_module_t) == 10 * sizeof(void *), "Unexpected native SCHIZO module layout");
_Static_assert(offsetof(orte_schizo_base_module_t, setup_fork) == 5 * sizeof(void *), "Unexpected native fork slot");

/** @brief Delegate only original native publication slots to the owned policy.
 * @param job Native ORTE job, unchanged by this adapter.
 * @param application Native application context, or null for fail-closed policy validation.
 * @return Exact native ORTE policy result.
 */
static int setup_fork(orte_job_t *job, orte_app_context_t *application)
{
    (void)job;
    return llaminar_mpi_native_setup_fork(orte_launch_environ,
        application ? &application->env : NULL);
}

/** @brief Native module table; the only installed callback is the existing fork edge. */
static orte_schizo_base_module_t module = {.setup_fork = setup_fork};

/** @brief Admit owned native daemons after the ordinary ompi personality.
 * @param result Native framework output slot for the exact module.
 * @param priority Native priority slot; zero retains the existing ordering.
 * @return Exact native selection/policy result, without application participation.
 */
static int query(mca_base_module_t **result, int *priority)
{
    *result = NULL;
    *priority = 0;
    if (ORTE_PROC_IS_APP) return ORTE_ERROR;
    const int status = llaminar_mpi_native_query_protocol();
    if (status == ORTE_SUCCESS) *result = (mca_base_module_t *)&module;
    return status;
}

/** @brief Native MCA discovery symbol bound to the selected genuine SDK ABI. */
__attribute__((visibility("default")))
orte_schizo_base_component_t mca_schizo_llaminar_component = {
    .base_version = {
        .mca_major_version = MCA_BASE_VERSION_MAJOR,
        .mca_minor_version = MCA_BASE_VERSION_MINOR,
        .mca_release_version = MCA_BASE_VERSION_RELEASE,
        .mca_project_name = "orte",
        .mca_project_major_version = ORTE_MAJOR_VERSION,
        .mca_project_minor_version = ORTE_MINOR_VERSION,
        .mca_project_release_version = ORTE_RELEASE_VERSION,
        .mca_type_name = "schizo",
        .mca_type_major_version = 1,
        .mca_type_minor_version = 0,
        .mca_type_release_version = 0,
        .mca_component_name = "llaminar",
        .mca_component_major_version = 1,
        .mca_query_component = query},
    .base_data = {.param_field = MCA_BASE_METADATA_PARAM_CHECKPOINT}};
