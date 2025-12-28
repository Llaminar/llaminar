# EnforceTypedRegisters.cmake
# 
# Build-time enforcement that JIT code uses the typed register system
# instead of raw Xbyak register instantiations.
#
# This prevents developers from bypassing the RegisterGuard system by
# directly constructing registers like Zmm(5) or Xmm(20).

# Files that are allowed to use raw register construction
# (these define the typed register infrastructure itself)
set(TYPED_REGISTER_INFRASTRUCTURE_FILES
    "RegisterAllocation.h"
    "RegisterGuard.h"
    "RegisterEnforcement.h"
    "JitMicrokernelBase.h"
)

# Legacy files that predate the typed register system
# TODO: Migrate these to use typed accessors (e.g., accum0().xmm() instead of Xmm(0))
# These files use Xmm(N) to access the low 128 bits of ZMM accumulators, which is
# technically correct (same physical register) but bypasses tracking.
set(TYPED_REGISTER_LEGACY_FILES
    "JitFusedAttentionWo.h"    # Uses Xmm(0-7) for accum aliasing, Xmm(16-19) for state
    "JitQ8DotProduct.h"        # Uses Xmm/Ymm for input/scratch zones
    "JitWoProjection.h"        # Uses Zmm/Reg64 for direct register access
    # Q16_1 JIT microkernels (Phase 6 scaffolding - TODO: migrate to borrow<>())
    "JitQ16DotProduct.h"       # Q×K^T dot product scaffolding
    "JitExp2FixedSoftmax.h"    # Exp2 softmax scaffolding
    "JitPVAccumulate.h"        # P×V accumulation scaffolding
    "JitWoProjectionVNNI.h"    # VPDPWSSD Wo projection scaffolding
    "JitQ16FusedAttention.h"   # Fused attention orchestrator scaffolding
)

# Function to check a single file for raw register violations
function(check_file_for_raw_registers FILE_PATH VIOLATIONS_VAR)
    # Read file content
    file(READ "${FILE_PATH}" FILE_CONTENT)
    
    # ==========================================================================
    # CATEGORY 1: Raw Xbyak register construction with numeric arguments
    # ==========================================================================
    # We need to catch multiple patterns:
    #   Direct: Zmm(5), Xbyak::Zmm(5)
    #   Variable: Zmm foo(5), Xbyak::Zmm bar(20)
    #   Assignment: auto x = Zmm(5)
    #
    # The key insight is that raw construction always has a numeric literal in parens
    # after the register class name, possibly with an identifier in between.
    #
    # Pattern breakdown:
    #   (Xbyak::)?     - Optional Xbyak:: prefix
    #   (Zmm|Xmm|Ymm|Reg64|Reg32)  - Register class
    #   [^(]*          - Anything except ( (captures variable name if present)
    #   \(             - Opening paren
    #   [ \t]*         - Optional whitespace
    #   [0-9]          - Numeric argument (the violation)
    set(RAW_REGISTER_PATTERNS
        "Zmm[^(]*\\([ \t]*[0-9]"
        "Xmm[^(]*\\([ \t]*[0-9]"
        "Ymm[^(]*\\([ \t]*[0-9]"
        "Reg64[^(]*\\([ \t]*[0-9]"
        "Reg32[^(]*\\([ \t]*[0-9]"
    )
    
    # ==========================================================================
    # CATEGORY 2: Raw typed accessor usage (bypasses borrow tracking)
    # ==========================================================================
    # These patterns catch direct use of typed accessors without borrow<>():
    #   BAD:  gen.scratch0().zmm()   - no RAII tracking
    #   GOOD: gen.borrow<Scratch0>() - RAII tracked with lifetime
    #
    # Pattern: .<accessor>().<reg_method>()
    # where accessor is accum0-7, scratch0-5, input0-7, state_*, score0-3, const_*
    # and reg_method is zmm, xmm, ymm
    set(RAW_ACCESSOR_PATTERNS
        # Accumulator zone (zmm0-7)
        "accum[0-7]\\(\\)\\.zmm"
        "accum[0-7]\\(\\)\\.xmm"
        "accum[0-7]\\(\\)\\.ymm"
        # Input/Q vector zone (zmm8-15)
        "input[0-7]\\(\\)\\.zmm"
        "input[0-7]\\(\\)\\.xmm"
        "input[0-7]\\(\\)\\.ymm"
        # State zone (zmm16-19)
        "state_max\\(\\)\\.zmm"
        "state_sum\\(\\)\\.zmm"
        "state_weight\\(\\)\\.zmm"
        "state_corr\\(\\)\\.zmm"
        # Scratch zone (zmm20-25)
        "scratch[0-5]\\(\\)\\.zmm"
        "scratch[0-5]\\(\\)\\.xmm"
        "scratch[0-5]\\(\\)\\.ymm"
        # Score zone (xmm20-23, aliases Scratch)
        "score[0-3]\\(\\)\\.xmm"
        "score[0-3]\\(\\)\\.zmm"
        # Constant zone (zmm26-31) - these are read-only, less critical but still flag
        "const_[a-z0-9_]+\\(\\)\\.zmm"
    )
    
    set(LOCAL_VIOLATIONS "")
    
    # Check raw register patterns (Category 1)
    foreach(PATTERN ${RAW_REGISTER_PATTERNS})
        string(REGEX MATCHALL "${PATTERN}" MATCHES "${FILE_CONTENT}")
        if(MATCHES)
            list(LENGTH MATCHES MATCH_COUNT)
            list(APPEND LOCAL_VIOLATIONS "RAW_REG: ${PATTERN} (${MATCH_COUNT})")
        endif()
    endforeach()
    
    # Check raw accessor patterns (Category 2)
    foreach(PATTERN ${RAW_ACCESSOR_PATTERNS})
        string(REGEX MATCHALL "${PATTERN}" MATCHES "${FILE_CONTENT}")
        if(MATCHES)
            list(LENGTH MATCHES MATCH_COUNT)
            list(APPEND LOCAL_VIOLATIONS "RAW_ACCESSOR: ${PATTERN} (${MATCH_COUNT})")
        endif()
    endforeach()
    
    set(${VIOLATIONS_VAR} "${LOCAL_VIOLATIONS}" PARENT_SCOPE)
endfunction()

# Main enforcement function - call this from CMakeLists.txt
function(enforce_typed_register_usage SOURCE_DIR)
    message(STATUS "Checking JIT files for raw Xbyak register usage...")
    
    # Find all header and source files in /jit/ directories
    # Note: CMake GLOB_RECURSE doesn't support ** globstar, so we use explicit depth patterns
    file(GLOB_RECURSE JIT_FILES LIST_DIRECTORIES false
        "${SOURCE_DIR}/kernels/*/jit/*.h"
        "${SOURCE_DIR}/kernels/*/jit/*.hpp"
        "${SOURCE_DIR}/kernels/*/jit/*.cpp"
        "${SOURCE_DIR}/kernels/*/*/jit/*.h"
        "${SOURCE_DIR}/kernels/*/*/jit/*.hpp"
        "${SOURCE_DIR}/kernels/*/*/jit/*.cpp"
        "${SOURCE_DIR}/kernels/*/*/*/jit/*.h"
        "${SOURCE_DIR}/kernels/*/*/*/jit/*.hpp"
        "${SOURCE_DIR}/kernels/*/*/*/jit/*.cpp"
        "${SOURCE_DIR}/kernels/*/*/*/*/jit/*.h"
        "${SOURCE_DIR}/kernels/*/*/*/*/jit/*.hpp"
        "${SOURCE_DIR}/kernels/*/*/*/*/jit/*.cpp"
    )
    
    set(VIOLATING_FILES "")
    set(VIOLATION_DETAILS "")
    
    foreach(FILE_PATH ${JIT_FILES})
        # Get just the filename
        get_filename_component(FILENAME "${FILE_PATH}" NAME)
        
        # Skip infrastructure files (define the typed register system)
        list(FIND TYPED_REGISTER_INFRASTRUCTURE_FILES "${FILENAME}" IS_INFRASTRUCTURE)
        if(NOT IS_INFRASTRUCTURE EQUAL -1)
            continue()
        endif()
        
        # Skip legacy files (predate typed register system, tracked for migration)
        list(FIND TYPED_REGISTER_LEGACY_FILES "${FILENAME}" IS_LEGACY)
        if(NOT IS_LEGACY EQUAL -1)
            continue()
        endif()
        
        # Check for violations
        check_file_for_raw_registers("${FILE_PATH}" FILE_VIOLATIONS)
        
        if(FILE_VIOLATIONS)
            # Get relative path for cleaner output
            file(RELATIVE_PATH REL_PATH "${SOURCE_DIR}" "${FILE_PATH}")
            list(APPEND VIOLATING_FILES "${REL_PATH}")
            list(APPEND VIOLATION_DETAILS "\n  ${REL_PATH}:")
            foreach(V ${FILE_VIOLATIONS})
                list(APPEND VIOLATION_DETAILS "\n    - ${V}")
            endforeach()
        endif()
    endforeach()
    
    # Report results
    list(LENGTH VIOLATING_FILES VIOLATION_COUNT)
    if(VIOLATION_COUNT GREATER 0)
        string(JOIN "" DETAILS_STR ${VIOLATION_DETAILS})
        message(FATAL_ERROR "
╔══════════════════════════════════════════════════════════════════════════════╗
║                 REGISTER GUARD BYPASS DETECTED                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ${VIOLATION_COUNT} file(s) bypass the RegisterGuard tracking system.                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VIOLATIONS FOUND:${DETAILS_STR}
╠══════════════════════════════════════════════════════════════════════════════╣
║  VIOLATION TYPES:                                                            ║
║                                                                              ║
║  RAW_REG: Direct Xbyak register construction with numeric index              ║
║      BAD:  Zmm zmm_temp(20);                                                 ║
║      BAD:  gen.vmovaps(Zmm(5), ...);                                         ║
║                                                                              ║
║  RAW_ACCESSOR: Typed accessor without borrow<>() tracking                    ║
║      BAD:  gen.scratch0().zmm();     // No lifetime tracking!                ║
║      BAD:  auto zmm = accum4().zmm(); // Bypasses guard system!              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  HOW TO FIX:                                                                 ║
║                                                                              ║
║  Always use borrow<>() for RAII-tracked register access:                     ║
║      auto guard = gen.borrow<Scratch0>();  // RAII tracked                   ║
║      gen.vmovaps(guard.zmm(), ...);        // Compiler-enforced lifetime     ║
║                                                                              ║
║  Available register types (see RegisterAllocation.h):                        ║
║      Accum0-7    - Accumulator zone (zmm0-7)                                 ║
║      Input0-7    - Input/Q vector zone (zmm8-15)                             ║
║      StateMax, StateSum, StateWeight, StateCorr - State zone (zmm16-19)      ║
║      Scratch0-5  - Scratch zone (zmm20-25)                                   ║
║      Score0-3    - Score zone (xmm20-23, aliases Scratch0-3)                 ║
║                                                                              ║
║  See: kernels/cpu/jit/RegisterGuard.h for documentation                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
")
    else()
        message(STATUS "  ✓ All JIT files use typed register system correctly")
    endif()
endfunction()
