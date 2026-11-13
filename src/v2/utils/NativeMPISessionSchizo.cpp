/**
 * @file NativeMPISessionSchizo.cpp
 * @brief Native OpenMPI fork hook retaining daemon-local owned runtime paths.
 *
 * OpenMPI's ordinary ompi personality merges launcher-valued application env
 * over daemon env. This additional native SCHIZO module runs afterwards and
 * projects only the shared ownership declaration from orte_launch_environ.
 * It creates no process/file owner, probes no devices and initializes no MPI.
 * Unowned native jobs remain unchanged. The exact supported ORTE/MCA ABI is
 * checked both by the native loader and the helper's mandatory module query.
 */
#include "utils/MPISessionEnvironment.h"
#include "orte_config.h"
#include "ompi/version.h"
#include "orte/version.h"
#include "orte/constants.h"
#include "orte/runtime/orte_globals.h"
#include "orte/mca/schizo/schizo.h"
#include "opal/util/opal_environ.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>

#if OMPI_MAJOR_VERSION!=4 || OMPI_MINOR_VERSION!=1 || OMPI_RELEASE_VERSION!=6
#error "Owned MPI native environment component requires the selected OpenMPI 4.1.6 headers"
#endif
static_assert(sizeof(orte_schizo_base_module_t)==10*sizeof(void*));
static_assert(offsetof(orte_schizo_base_module_t,setup_fork)==5*sizeof(void*));
extern "C" { extern const char orte_version_string[]; }

namespace
{
    using namespace llaminar2::mpi_session;
    /** @return The original native daemon value without consulting launcher app->env. */
    const char *value(char **environment,const char *name)
    {
        const auto length=std::strlen(name);
        for(char **entry=environment;entry && *entry;++entry)
            if(std::strncmp(*entry,name,length)==0 && (*entry)[length]=='=')return *entry+length+1;
        return nullptr;
    }
    /** @return Whether the actually loaded native runtime matches this compiled ABI. */
    bool compatible() { return std::strcmp(orte_version_string,"4.1.6")==0; }
    /** @brief Fail before native rank creation with an ownership-specific diagnostic. */
    int reject(const char *reason)
    {
        std::fprintf(stderr,"Owned MPI native environment publication failed: %s\n",reason);
        return ORTE_ERR_BAD_PARAM;
    }
    /** @brief Project only original daemon-owned paths after ompi's normal env merge. */
    int setupFork(orte_job_t *job,orte_app_context_t *application)
    {
        (void)job;
        const char *required=value(orte_launch_environ,kRequiredEnv);
        if(!required)return ORTE_SUCCESS;
        if(!compatible() || std::strcmp(required,kProtocol)!=0)return reject("unsupported native protocol/ABI");
        if(!application)return reject("missing native application context");
        for(const auto &binding:kLocationBindings) {
            const char *original=value(orte_launch_environ,binding.name);
            if(!original || !*original)return reject("daemon has no original owned location binding");
            if(opal_setenv(binding.name,original,true,&application->env)!=OPAL_SUCCESS)
                return reject("cannot publish original daemon location");
        }
        for(const char *name:kUnsetLocations) {
            if(value(orte_launch_environ,name))return reject("daemon has conflicting native location aliases");
            const int result=opal_unsetenv(name,&application->env);
            if(result!=OPAL_SUCCESS && result!=OPAL_ERR_NOT_FOUND)return reject("cannot retire conflicting app location alias");
        }
        for(const char *name:kProjectedControls) {
            const char *original=value(orte_launch_environ,name);
            if(!original || !*original)return reject("daemon has no original native component binding");
            if(opal_setenv(name,original,true,&application->env)!=OPAL_SUCCESS)
                return reject("cannot publish original native component binding");
        }
        try {
            const auto ready=rootReadiness(value(orte_launch_environ,"TMPDIR"),
                value(orte_launch_environ,"OMPI_MCA_btl_vader_backing_directory"));
            if(opal_setenv(kReadyEnv,ready.c_str(),true,&application->env)!=OPAL_SUCCESS)
                return reject("cannot publish native rank readiness");
        } catch(const std::exception &error) { return reject(error.what()); }
        return ORTE_SUCCESS;
    }
    /** @brief Native module table; no callback except the existing pre-fork environment hook. */
    orte_schizo_base_module_t module={nullptr,nullptr,nullptr,nullptr,nullptr,setupFork,nullptr,nullptr,nullptr,nullptr};
    /** @brief Participate only for owned native tools/daemons, after ompi priority10. */
    int query(mca_base_module_t **result,int *priority)
    {
        *result=nullptr;*priority=0;
        if(ORTE_PROC_IS_APP)return ORTE_ERROR;
        const char *required=processEnvironmentValue(EnvironmentKey::RequiredProtocol);
        if(!required)return ORTE_ERROR;
        if(!compatible() || std::strcmp(required,kProtocol)!=0)return reject("incompatible selected native component");
        *result=reinterpret_cast<mca_base_module_t*>(&module);
        return ORTE_SUCCESS;
    }
    /** @return Zero-initialized component metadata bound to the exact native headers. */
    orte_schizo_base_component_t component()
    {
        orte_schizo_base_component_t result{};
        auto &base=result.base_version;
        base.mca_major_version=MCA_BASE_VERSION_MAJOR;
        base.mca_minor_version=MCA_BASE_VERSION_MINOR;
        base.mca_release_version=MCA_BASE_VERSION_RELEASE;
        std::strcpy(base.mca_project_name,"orte");
        base.mca_project_major_version=ORTE_MAJOR_VERSION;
        base.mca_project_minor_version=ORTE_MINOR_VERSION;
        base.mca_project_release_version=ORTE_RELEASE_VERSION;
        std::strcpy(base.mca_type_name,"schizo");
        base.mca_type_major_version=1;base.mca_type_minor_version=0;base.mca_type_release_version=0;
        std::strcpy(base.mca_component_name,"llaminar");
        base.mca_component_major_version=1;
        base.mca_query_component=query;
        result.base_data.param_field=MCA_BASE_METADATA_PARAM_CHECKPOINT;
        return result;
    }
}
extern "C"
{
    /** @return Pure ABI metadata only when the actually loaded native SDK matches. */
    __attribute__((visibility("default"))) const llaminar2::mpi_session::NativeModuleContract *llaminar_mpi_native_contract_v1()
    {
        static const llaminar2::mpi_session::NativeModuleContract contract;
        return compatible()?&contract:nullptr;
    }
    /** @brief Native MCA discovery symbol; selected by OpenMPI's existing multi-module framework. */
    __attribute__((visibility("default"))) orte_schizo_base_component_t mca_schizo_llaminar_component=component();
}
