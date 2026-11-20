/**
 * @file CUDADiscoveryAdmission.cpp
 * @brief Single native loader probe behind the typed CUDA discovery boundary.
 *
 * One implementation answers the loader question for both the production
 * driver-presence authority (CUDADriverApi::probeDriverBinding, which owns the
 * exact libcuda.so.1 request) and the device-free admission proof. The flags
 * match the authoritative binding exactly, so a presence answer cannot come
 * from a lazier load or a different search-path question.
 */
#include "CUDADiscoveryAdmission.h"

#include <dlfcn.h>
#include <string>

namespace llaminar2
{
    CUDADriverLoaderOutcome probeNativeLibraryBinding(std::string_view request) noexcept
    {
        const std::string terminated{request};
        (void)dlerror(); // Clear only the loader's thread-local diagnostic.
        if (void *handle = dlopen(terminated.c_str(), RTLD_NOW | RTLD_LOCAL))
        {
            // Nothing was called through the handle; an already loaded object
            // only gains and loses one reference count.
            dlclose(handle);
            return {true, {}};
        }
        const char *error = dlerror();
        return {false, error != nullptr ? std::string{error} : std::string{}};
    }
}
