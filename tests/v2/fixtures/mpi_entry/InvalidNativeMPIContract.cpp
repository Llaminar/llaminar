/**
 * @file InvalidNativeMPIContract.cpp
 * @brief Typed incompatible native-module contracts for installed preflight.
 *
 * The fixture uses the same C++17 contract declaration as the actual helper,
 * but changes exactly one required version/layout fact. It requires no OpenMPI
 * headers or runtime. The helper must emit its precise ABI diagnostic before
 * launcher/rank creation; a crash or arbitrary failure never certifies it.
 */
#include "utils/MPISessionEnvironment.h"

namespace
{
    /** @return One changed immutable contract fact with every other field valid. */
    llaminar2::mpi_session::NativeModuleContract incompatibleContract()
    {
        llaminar2::mpi_session::NativeModuleContract contract;
#ifdef LLAMINAR_MPI_INVALID_VERSION
        ++contract.openmpi_release;
#else
        contract.schizo_module_bytes += sizeof(void *);
#endif
        return contract;
    }
}

/**
 * @brief Export the same native entrypoint with an incompatible descriptor.
 * @return A stable immutable descriptor that the helper must reject precisely.
 */
extern "C" const llaminar2::mpi_session::NativeModuleContract *llaminar_mpi_native_contract_v1()
{
    static const auto contract = incompatibleContract();
    return &contract;
}
