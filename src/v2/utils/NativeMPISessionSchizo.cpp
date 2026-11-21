/**
 * @file NativeMPISessionSchizo.cpp
 * @brief Owned environment policy behind the native OpenMPI C component adapter.
 *
 * ORTE structures/internal atomic headers are consumed only by the genuine C
 * adapter. This C++17 policy retains the one typed environment schema and
 * readiness authority. It creates no process/file owner, probes no devices and
 * initializes no MPI. Unowned jobs remain unchanged; exact runtime ABI and
 * daemon-local values are still mandatory before native rank publication.
 */
#include "utils/MPISessionEnvironment.h"
#include "utils/NativeMPISessionSchizoBridge.h"
#include "orte_config.h"
#include "ompi/version.h"
#include "orte/version.h"
#include "orte/constants.h"
#include "opal/util/opal_environ.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if OMPI_MAJOR_VERSION != 4 || OMPI_MINOR_VERSION != 1 || OMPI_RELEASE_VERSION != 6
#error "Owned MPI native environment component requires the selected OpenMPI 4.1.6 headers"
#endif
extern "C" { extern const char orte_version_string[]; }

namespace
{
    /** @brief Find an original daemon binding without reading the merged application.
     * @param environment Borrowed original environment.
     * @param name Source-declared owned binding.
     * @return Borrowed value or null, preserving unset versus empty.
     */
    const char *value(char **environment, const char *name)
    {
        const auto length = std::strlen(name);
        for (char **entry = environment; entry && *entry; ++entry)
            if (std::strncmp(*entry, name, length) == 0 && (*entry)[length] == '=')
                return *entry + length + 1;
        return nullptr;
    }
    /** @return Whether the actually loaded runtime matches the compiled native ABI. */
    bool compatible() { return std::strcmp(orte_version_string, "4.1.6") == 0; }
    /** @brief Reject publication before native rank creation.
     * @param reason Precise failed ownership/protocol invariant.
     * @return Native ORTE bad-parameter code.
     */
    int reject(const char *reason)
    {
        std::fprintf(stderr, "Owned MPI native environment publication failed: %s\n", reason);
        return ORTE_ERR_BAD_PARAM;
    }
}

extern "C" int llaminar_mpi_native_setup_fork(char **daemon_environment, char ***application_environment)
{
    using namespace llaminar2::mpi_session;
    const char *required = value(daemon_environment, kRequiredEnv);
    if (!required) return ORTE_SUCCESS;
    if (!compatible() || std::strcmp(required, kProtocol) != 0)
        return reject("unsupported native protocol/ABI");
    if (!application_environment) return reject("missing native application context");
    for (const auto &binding : kLocationBindings)
    {
        const char *original = value(daemon_environment, binding.name);
        if (!original || !*original) return reject("daemon has no original owned location binding");
        if (opal_setenv(binding.name, original, true, application_environment) != OPAL_SUCCESS)
            return reject("cannot publish original daemon location");
    }
    for (const char *name : kUnsetLocations)
    {
        if (value(daemon_environment, name)) return reject("daemon has conflicting native location aliases");
        const int result = opal_unsetenv(name, application_environment);
        if (result != OPAL_SUCCESS && result != OPAL_ERR_NOT_FOUND)
            return reject("cannot retire conflicting app location alias");
    }
    for (const char *name : kProjectedControls)
    {
        const char *original = value(daemon_environment, name);
        if (!original || !*original) return reject("daemon has no original native component binding");
        if (opal_setenv(name, original, true, application_environment) != OPAL_SUCCESS)
            return reject("cannot publish original native component binding");
    }
    try
    {
        const auto ready = rootReadiness(value(daemon_environment, "TMPDIR"),
            value(daemon_environment, "OMPI_MCA_btl_vader_backing_directory"));
        if (opal_setenv(kReadyEnv, ready.c_str(), true, application_environment) != OPAL_SUCCESS)
            return reject("cannot publish native rank readiness");
    }
    catch (const std::exception &error) { return reject(error.what()); }
    return ORTE_SUCCESS;
}

extern "C" int llaminar_mpi_native_query_protocol(void)
{
    using namespace llaminar2::mpi_session;
    const char *required = processEnvironmentValue(EnvironmentKey::RequiredProtocol);
    if (!required) return ORTE_ERROR;
    if (!compatible() || std::strcmp(required, kProtocol) != 0)
        return reject("incompatible selected native component");
    return ORTE_SUCCESS;
}

/** @return Pure ABI metadata only when the actually loaded native SDK matches. */
extern "C" __attribute__((visibility("default")))
const llaminar2::mpi_session::NativeModuleContract *llaminar_mpi_native_contract_v1()
{
    static const llaminar2::mpi_session::NativeModuleContract contract;
    return compatible() ? &contract : nullptr;
}
