/**
 * @file CUDADriverApi.cpp
 * @brief Resolve required CUDA driver symbols at CUDA preparation, never ELF startup.
 *
 * POSIX resolves the exact symbol selected by the installed cuda.h declaration.
 * This preserves ABI versions (notably graph parameters and stream memory ops)
 * without hand-written signatures or guessed toolkit-to-function versions.
 * Incomplete construction releases its library reference and throws; a complete
 * table pins native code for the lifetime of cudart and all retained graphs.
 */
#include "CUDADriverApi.h"

#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace llaminar2
{
namespace
{
/** @brief Release only a failed binding attempt, never a published table. */
struct DriverLibraryCloser
{
    /** @brief Drop this factory's reference while unwinding incomplete binding. */
    void operator()(void *library) const noexcept { (void)dlclose(library); }
};

/**
 * @brief Resolve one mandatory symbol with the pointer type declared by cuda.h.
 * @param library Exact libcuda.so.1 handle retained by the table factory.
 * @param symbol Fully expanded SDK symbol, including its ABI/stream suffix.
 * @return Non-null native function pointer; failure never selects another ABI.
 */
template <typename Function>
Function requireDriverFunction(void *library, const char *symbol)
{
    (void)dlerror(); // Clear only the loader's thread-local diagnostic.
    void *address = dlsym(library, symbol);
    const char *error = dlerror();
    if (error || !address)
        throw std::runtime_error(
            std::string("CUDA driver is missing required SDK entry point ") +
            symbol + ": " + (error ? error : "null function address"));
    return reinterpret_cast<Function>(address);
}
} // namespace

const CUDADriverApi &CUDADriverApi::instance()
{
    static const CUDADriverApi api = []
    {
        // No toolkit stub, alternate driver path, runtime initialization or
        // device allocation is allowed here. Normal deployment owns libcuda.
        std::unique_ptr<void, DriverLibraryCloser> library(
            dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL));
        if (!library)
        {
            const char *error = dlerror();
            throw std::runtime_error(
                std::string("CUDA execution requires host NVIDIA driver libcuda.so.1: ") +
                (error ? error : "loader returned no handle"));
        }
        const CUDADriverApi complete(library.get());
        // CUDA runtime/static owners can outlive this table's destructor. The
        // OS releases this one reference at process exit, not model teardown.
        (void)library.release();
        return complete;
    }();
    return api;
}

#define LLAMINAR_CUDA_DRIVER_SYMBOL_TEXT_(symbol) #symbol
#define LLAMINAR_CUDA_DRIVER_SYMBOL_TEXT(symbol) LLAMINAR_CUDA_DRIVER_SYMBOL_TEXT_(symbol)
CUDADriverApi::CUDADriverApi(void *library)
    : library_(library)
#define LLAMINAR_CUDA_DRIVER_ENTRY(member, symbol) \
    , member(requireDriverFunction<decltype(&::symbol)>( \
        library_, LLAMINAR_CUDA_DRIVER_SYMBOL_TEXT(symbol)))
#include "CUDADriverFunctions.def"
#undef LLAMINAR_CUDA_DRIVER_ENTRY
{
}
#undef LLAMINAR_CUDA_DRIVER_SYMBOL_TEXT
#undef LLAMINAR_CUDA_DRIVER_SYMBOL_TEXT_
} // namespace llaminar2
