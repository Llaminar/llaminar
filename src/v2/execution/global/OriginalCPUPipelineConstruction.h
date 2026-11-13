/**
 * @file OriginalCPUPipelineConstruction.h
 * @brief Preserve original cold-construction errors before retained CPU native duties unwind.
 *
 * A domain registry is outside this guarded body. Once it owns admitted CPU
 * domains, a later child/source/PMA failure cannot safely close them locally:
 * original peers may still occupy a different construction collective. Report
 * the original cause and abort the SAME parent communicator while those strict
 * native owners remain alive. No destructor is weakened and no MPI group is
 * reconstructed. Non-native construction retains ordinary exception semantics.
 */
#pragma once
#include "DomainCommunicatorRegistry.h"
#include "interfaces/IMPIContext.h"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <utility>
namespace llaminar2 {
/**
 * @brief Execute original domain/child construction without losing its first fatal diagnostic.
 * @tparam Build Actual caller-owned original construction body.
 * @param domains Sole original registry, outliving this boundary and all body temporaries.
 * @param parent Exact original parent context used by that registry's native splits.
 * @param build Same original setup body; no alternative factory or execution path.
 * @return Original successful construction result unchanged.
 * @throws Original exception only before any CPU native-duty registry exists.
 * @note A failed native CPU body aborts its original parent; returning from MPI_Abort is fatal.
 */
template<class Build>
decltype(auto) withOriginalCPUPipelineConstruction(DomainCommunicatorRegistry& domains,
    const IMPIContext& parent,Build&& build) {
    const auto abort_original=[&](const char* reason) {
        if(!domains.originalCPUOwners())return;
        // C stdio preserves the first failure even if the exception was an
        // allocation failure. Flush before MPI abort can terminate this rank.
        std::fprintf(stderr,"[ERROR] [OriginalCPUPipelineConstruction] rank=%d/%d reason=%s; "
            "aborting original parent communicator before native CPU owner unwind\n",
            parent.rank(),parent.world_size(),reason);
        std::fflush(stderr);
        const auto original=parent.communicator();
        if(original!=MPI_COMM_NULL)(void)MPI_Abort(original,EXIT_FAILURE);
        std::terminate();
    };
    try {return std::forward<Build>(build)();}
    catch(const std::exception& error) {abort_original(error.what());throw;}
    catch(...) {abort_original("non-standard construction exception");throw;}
}
}
