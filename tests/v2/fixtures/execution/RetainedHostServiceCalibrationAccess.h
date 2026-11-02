/** @file RetainedHostServiceCalibrationAccess.h
 * @brief Explicit unqualified actual-program diagnostic bridge; no installed limit or production override.
 */
#pragma once
#if !defined(LLAMINAR_HOST_SERVICE_CALIBRATION) && !defined(LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC)
#error "CPU service unqualified diagnostic access belongs only to explicit raw native program targets"
#endif
#include "execution/local_execution/graph/RetainedHostServiceDeclarations.h"
namespace llaminar2::test {
/** @brief Constructs only tagged unqualified declarations; native creation uses the ordinary owner body. */
struct RetainedHostServiceCalibrationAccess final {
    /** @brief Authenticate immutable original provider/TLS/default facts before a diagnostic lazy-work boundary. */
    static void requireCurrent(const RetainedHostServiceProgram& program){program.requireCurrent();}
    /** @return Actual metadata/key/geometry, with no libc/TLS allowance or qualification. */
    /** @return Same raw evidence role using the one original declaration session, without rehash. */
    static RetainedHostServiceProgram program(RetainedHostServiceRole role,std::shared_ptr<const RetainedHostServiceProgram::Evidence> evidence){return RetainedHostServiceProgram::snapshot(role,RetainedHostServiceProgram::MetadataAuthority::UnqualifiedDiagnostic,std::move(evidence));}
    static RetainedHostServiceProgram program(RetainedHostServiceRole role){return RetainedHostServiceProgram::unqualifiedDiagnostic(role);}
};
}
