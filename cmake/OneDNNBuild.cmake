# @file OneDNNBuild.cmake
# @brief Own one per-ISA oneDNN build through separate configured/installed states.
#
# An interrupted compilation is not a stale configuration. Retain its proven
# objects so the next unrestricted build resumes, but never advertise an install
# until the complete build and install phases succeed. Durable retirement intent
# survives interrupted invalidation; a benign failed compilation keeps its work.
# Identity binds source, compilers,
# flags, generator and Ninja authority, not only an ISA label. The phase owner is
# separately testable with a custom-command-only fixture: installed tests need
# neither a compiler SDK nor a counterfeit oneDNN implementation.
include_guard(GLOBAL)

# @brief Authenticate an existing source without mutating or deleting it.
# @param git Executable selected by the caller's real Git discovery.
# @param source Existing oneDNN checkout, which remains untouched on failure.
# @param expected_origin Canonical upstream URL owned by dependency selection.
function(llaminar_require_onednn_origin git source expected_origin)
    execute_process(COMMAND "${git}" -C "${source}" remote get-url origin
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
    if(NOT result STREQUAL "0")
        message(FATAL_ERROR "oneDNN origin authentication failed (${result}) at ${source}\nstdout:\n${output}\nstderr:\n${error}\nExisting source left untouched.")
    endif()
    if(NOT output STREQUAL expected_origin)
        message(FATAL_ERROR "oneDNN origin mismatch at ${source}: observed '${output}', required '${expected_origin}'. Existing source left untouched; select an authenticated checkout explicitly.")
    endif()
endfunction()

# @brief Authenticate the entire installed manifest, not just one public header.
# @param build Owned installation/build prefix.
# @param digest_out Parent variable receiving a content-bound closure digest, or empty.
# @param reason_out Parent variable receiving the first incomplete-closure diagnostic.
# @param library_out Parent variable receiving the exact manifested consumer library.
function(_llaminar_onednn_install_digest build digest_out reason_out library_out)
    set(${digest_out} "" PARENT_SCOPE)
    set(${library_out} "" PARENT_SCOPE)
    set(${reason_out} "missing install manifest/public header/library" PARENT_SCOPE)
    if(NOT EXISTS "${build}/install_manifest.txt" OR
       NOT EXISTS "${build}/include/dnnl.hpp" OR
       NOT EXISTS "${build}/include/oneapi/dnnl/dnnl.hpp")
        return()
    endif()
    file(STRINGS "${build}/install_manifest.txt" artifacts)
    if(NOT artifacts)
        return()
    endif()
    get_filename_component(prefix "${build}" REALPATH)
    set(evidence)
    set(selected_library)
    foreach(artifact IN LISTS artifacts)
        if(NOT EXISTS "${artifact}" OR IS_DIRECTORY "${artifact}")
            set(${reason_out} "missing installed artifact: ${artifact}" PARENT_SCOPE)
            return()
        endif()
        get_filename_component(actual "${artifact}" REALPATH)
        string(FIND "${actual}" "${prefix}/" inside)
        if(NOT inside EQUAL 0)
            set(${reason_out} "installed artifact escapes owned prefix: ${artifact}" PARENT_SCOPE)
            return()
        endif()
        if(artifact MATCHES "/lib(64)?/libdnnl\\.(so(\\.[0-9]+)*|dylib|a)$")
            if(NOT selected_library)
                set(selected_library "${artifact}")
            endif()
        endif()
        file(SHA256 "${artifact}" artifact_hash)
        string(APPEND evidence "${artifact}=${artifact_hash}\n")
    endforeach()
    if(NOT selected_library)
        return()
    endif()
    file(SHA256 "${build}/install_manifest.txt" manifest_hash)
    string(SHA256 digest "${manifest_hash}\n${evidence}")
    set(${digest_out} "${digest}" PARENT_SCOPE)
    set(${reason_out} "" PARENT_SCOPE)
    set(${library_out} "${selected_library}" PARENT_SCOPE)
endfunction()

# @brief Serialize mandatory parent compiler/toolchain arguments without invoking an SDK.
# @param args_out Parent variable receiving the exact child CMake argument list.
# @param identity_out Parent variable receiving the same fields for cache identity.
# @details ARG1 belongs to the compiler invocation, not an optional launcher.
function(_llaminar_onednn_toolchain_arguments args_out identity_out)
    set(arguments)
    set(identity)
    foreach(name IN ITEMS CMAKE_C_COMPILER_ARG1 CMAKE_CXX_COMPILER_ARG1
            CMAKE_TOOLCHAIN_FILE CMAKE_SYSROOT CMAKE_OSX_SYSROOT
            CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET
            CMAKE_C_COMPILER_TARGET CMAKE_CXX_COMPILER_TARGET
            CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN)
        if(DEFINED ${name} AND NOT "${${name}}" STREQUAL "")
            list(APPEND arguments "-D${name}=${${name}}")
            string(APPEND identity "${name}=${${name}}\n")
            if(name STREQUAL "CMAKE_TOOLCHAIN_FILE")
                file(SHA256 "${CMAKE_TOOLCHAIN_FILE}" toolchain_hash)
                string(APPEND identity "toolchain_hash=${toolchain_hash}\n")
            endif()
        endif()
    endforeach()
    set(${args_out} "${arguments}" PARENT_SCOPE)
    set(${identity_out} "${identity}" PARENT_SCOPE)
endfunction()

# @brief Reject unsafe destination ancestry before any producer can follow it.
# @param build Owned build/install prefix.
# @param wanted Identity written as durable retirement intent on invalid storage.
# @details GLOB_RECURSE does not follow directory links (CMP0009 NEW). In-prefix
# versioned library-file symlinks remain legal; directory/escaping/dangling links
# do not authorize writes, even without any earlier installation receipt.
function(_llaminar_check_onednn_surfaces build wanted)
    get_filename_component(prefix "${build}" REALPATH)
    foreach(surface IN ITEMS include lib lib64 share)
        set(root "${build}/${surface}")
        set(unsafe)
        if(IS_SYMLINK "${root}" OR (EXISTS "${root}" AND NOT IS_DIRECTORY "${root}"))
            set(unsafe "${root}")
        elseif(EXISTS "${root}")
            cmake_policy(PUSH)
            cmake_policy(SET CMP0009 NEW)
            file(GLOB_RECURSE entries LIST_DIRECTORIES TRUE "${root}/*")
            cmake_policy(POP)
            foreach(entry IN LISTS entries)
                if(IS_SYMLINK "${entry}")
                    if(IS_DIRECTORY "${entry}" OR NOT EXISTS "${entry}" OR
                       NOT surface MATCHES "^lib(64)?$")
                        set(unsafe "${entry}")
                    else()
                        get_filename_component(actual "${entry}" REALPATH)
                        string(FIND "${actual}" "${prefix}/${surface}/" inside)
                        if(NOT inside EQUAL 0)
                            set(unsafe "${entry}")
                        endif()
                    endif()
                    if(unsafe)
                        break()
                    endif()
                endif()
            endforeach()
        endif()
        if(unsafe)
            file(WRITE "${build}/.llaminar-onednn-retirement-needed" "${wanted}\n")
            message(FATAL_ERROR "oneDNN install path is not an owned directory/file: ${unsafe}")
        endif()
    endforeach()
endfunction()

# @brief Execute the single configured -> installed lifecycle for an owned tree.
# @param source Authenticated source directory supplied by the oneDNN adapter.
# @param build Owned per-ISA output directory, never a public/system install.
# @param identity Complete immutable configuration identity; hashed here once.
# @param library_out Parent variable receiving the exact authenticated link input.
# @param ARGN Exact additional CMake configuration arguments.
# @details Private phase authority. Tests exercise these actual operations with
# a LANGUAGES NONE fixture, not a second lifecycle or a fake compiler/SDK.
function(_llaminar_install_onednn source build identity library_out)
    if(IS_SYMLINK "${build}")
        message(FATAL_ERROR "oneDNN build prefix is not an owned directory: ${build}")
    endif()
    file(LOCK "${build}.llaminar.lock" GUARD FUNCTION TIMEOUT 30 RESULT_VARIABLE lock_result)
    if(NOT lock_result STREQUAL "0")
        message(FATAL_ERROR "oneDNN build ownership unavailable: ${build}: ${lock_result}")
    endif()
    string(SHA256 wanted "${identity}")
    set(configured_marker "${build}/.llaminar-onednn-configured")
    set(installed_marker "${build}/.llaminar-onednn-commit")
    set(artifacts_marker "${build}/.llaminar-onednn-installed-artifacts")
    set(retirement_marker "${build}/.llaminar-onednn-retirement-needed")
    set(configured "missing")
    if(EXISTS "${configured_marker}")
        file(READ "${configured_marker}" configured)
        string(STRIP "${configured}" configured)
    endif()
    if(EXISTS "${build}" AND NOT configured STREQUAL wanted)
        message(STATUS "V2: Invalidating unproven/different oneDNN configuration: ${build}")
        file(REMOVE_RECURSE "${build}")
    endif()
    set(installed "missing")
    set(incomplete_reason)
    if(EXISTS "${installed_marker}")
        file(READ "${installed_marker}" installed)
        string(STRIP "${installed}" installed)
    endif()
    if(installed STREQUAL wanted AND NOT EXISTS "${retirement_marker}")
        _llaminar_onednn_install_digest("${build}" actual_artifacts incomplete_reason selected_library)
        set(recorded_artifacts "missing")
        if(EXISTS "${artifacts_marker}")
            file(READ "${artifacts_marker}" recorded_artifacts)
            string(STRIP "${recorded_artifacts}" recorded_artifacts)
        endif()
        if(actual_artifacts AND actual_artifacts STREQUAL recorded_artifacts)
            message(STATUS "V2: Reusing authenticated oneDNN installation: ${build}")
            set(${library_out} "${selected_library}" PARENT_SCOPE)
            return()
        endif()
    endif()
    if(EXISTS "${installed_marker}" OR EXISTS "${artifacts_marker}")
        if(NOT incomplete_reason)
            set(incomplete_reason "stale receipt or installed manifest/content seal mismatch")
        endif()
        message(STATUS "V2: oneDNN install closure needs completion: ${incomplete_reason}")
        # Durable intent precedes receipt removal. A crash, rejected symlink or
        # partial retirement must not let the next invocation re-sign remnants.
        file(WRITE "${retirement_marker}" "${wanted}\n")
    endif()
    file(REMOVE "${installed_marker}" "${artifacts_marker}")
    # Validate every surface BEFORE every configure/install attempt, including
    # ordinary failed-compilation retries with no installed receipt at all.
    _llaminar_check_onednn_surfaces("${build}" "${wanted}")
    if(EXISTS "${retirement_marker}")
        # Size/mtime can make file(INSTALL) preserve changed bytes. Retire the
        # complete install surfaces, but retain src/CMakeFiles compilation work.
        foreach(surface IN ITEMS include lib lib64 share)
            file(REMOVE_RECURSE "${build}/${surface}")
            if(EXISTS "${build}/${surface}" OR IS_SYMLINK "${build}/${surface}")
                message(FATAL_ERROR "oneDNN could not retire invalid install surface: ${build}/${surface}")
            endif()
        endforeach()
        file(REMOVE "${retirement_marker}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${source}" -B "${build}" -G "${CMAKE_GENERATOR}"
            "-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}"
            "-DCMAKE_INSTALL_PREFIX=${build}" ${ARGN}
        RESULT_VARIABLE configure_result OUTPUT_VARIABLE configure_output ERROR_VARIABLE configure_error)
    if(NOT configure_result STREQUAL "0")
        message(FATAL_ERROR "oneDNN configure failed (${configure_result})\nstdout:\n${configure_output}\nstderr:\n${configure_error}")
    endif()
    # Configuration success authorizes reuse of objects, not an installation.
    file(WRITE "${configured_marker}" "${wanted}\n")
    file(WRITE "${build}/.llaminar-onednn-identity" "${identity}")
    _llaminar_check_onednn_surfaces("${build}" "${wanted}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${build}" --parallel
        RESULT_VARIABLE build_result OUTPUT_VARIABLE build_output ERROR_VARIABLE build_error)
    if(NOT build_result STREQUAL "0")
        message(FATAL_ERROR "oneDNN build failed (${build_result})\nstdout:\n${build_output}\nstderr:\n${build_error}")
    endif()
    _llaminar_check_onednn_surfaces("${build}" "${wanted}")
    execute_process(COMMAND "${CMAKE_COMMAND}" --install "${build}"
        RESULT_VARIABLE install_result OUTPUT_VARIABLE install_output ERROR_VARIABLE install_error)
    if(NOT install_result STREQUAL "0")
        file(WRITE "${retirement_marker}" "${wanted}\n")
        message(FATAL_ERROR "oneDNN install failed (${install_result})\nstdout:\n${install_output}\nstderr:\n${install_error}")
    endif()
    _llaminar_onednn_install_digest("${build}" installed_artifacts incomplete_reason selected_library)
    if(NOT installed_artifacts)
        file(WRITE "${retirement_marker}" "${wanted}\n")
        message(FATAL_ERROR "oneDNN install returned success with an incomplete required closure: ${incomplete_reason}: ${build}")
    endif()
    file(WRITE "${artifacts_marker}" "${installed_artifacts}\n")
    file(WRITE "${installed_marker}" "${wanted}\n")
    set(${library_out} "${selected_library}" PARENT_SCOPE)
endfunction()

# @brief Build the canonical pinned dependency with the parent native toolchain.
# @param source_dir Authenticated checkout/archive selected by the caller.
# @param source_id Complete revision supplied by the source authority.
# @param output_dir Parent-scope variable receiving the per-ISA installation path.
# @param output_library Parent-scope variable receiving the exact manifested link input.
function(llaminar_prepare_onednn source_dir source_id output_dir output_library)
    if(NOT CMAKE_GENERATOR STREQUAL "Ninja" OR NOT CMAKE_MAKE_PROGRAM)
        message(FATAL_ERROR "oneDNN requires the configured project Ninja executable")
    endif()
    set(supported_isas AVX2 AVX512 NEON NEON_DOTPROD NEON_I8MM)
    if(NOT LLAMINAR_CPU_ISA_UPPER IN_LIST supported_isas OR NOT source_id)
        message(FATAL_ERROR "oneDNN requires an authenticated native ISA and source identity")
    endif()
    get_filename_component(source "${source_dir}" REALPATH)
    if(NOT EXISTS "${source}/CMakeLists.txt")
        message(FATAL_ERROR "oneDNN source has no CMakeLists.txt: ${source}")
    endif()
    string(TOLOWER "${LLAMINAR_CPU_ISA_UPPER}" suffix)
    set(build "${source}/build-${suffix}")
    file(SHA256 "${CMAKE_C_COMPILER}" c_compiler_hash)
    file(SHA256 "${CMAKE_CXX_COMPILER}" cxx_compiler_hash)
    file(SHA256 "${CMAKE_MAKE_PROGRAM}" ninja_hash)
    # Old missing-receipt states did not retain retirement intent. They cannot
    # be distinguished from a benign compile failure, so migrate that older
    # lifecycle schema once rather than trusting an ambiguous prior state.
    set(identity "schema=3\nsource=${source}\nrevision=${source_id}\nisa=${LLAMINAR_CPU_ISA_UPPER}\nflags=${LLAMINAR_CPU_COMPILE_FLAGS}\ncc=${CMAKE_C_COMPILER};${CMAKE_C_COMPILER_ID};${CMAKE_C_COMPILER_VERSION};${c_compiler_hash}\ncxx=${CMAKE_CXX_COMPILER};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${cxx_compiler_hash}\nlauncher=${CMAKE_C_COMPILER_LAUNCHER};${CMAKE_CXX_COMPILER_LAUNCHER}\ngenerator=${CMAKE_GENERATOR}\nmake=${CMAKE_MAKE_PROGRAM};${ninja_hash}\ntype=Release\ncpu_runtime=OMP\nexperimental_ukernel=ON\n")
    _llaminar_onednn_toolchain_arguments(toolchain_args toolchain_identity)
    string(APPEND identity "${toolchain_identity}")
    _llaminar_install_onednn("${source}" "${build}" "${identity}" selected_library
        "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
        "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
        -DCMAKE_BUILD_TYPE=Release -DDNNL_CPU_RUNTIME=OMP
        -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_EXPERIMENTAL_UKERNEL=ON
        "-DCMAKE_CXX_FLAGS=${LLAMINAR_CPU_COMPILE_FLAGS}"
        "-DCMAKE_C_FLAGS=${LLAMINAR_CPU_COMPILE_FLAGS}" ${toolchain_args})
    set(${output_dir} "${build}" PARENT_SCOPE)
    set(${output_library} "${selected_library}" PARENT_SCOPE)
endfunction()
