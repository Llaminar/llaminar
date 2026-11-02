# Fingerprint native creator source and retain one normalized configure dependency.
# CMake preserves path spelling in its generated missing-input phony rules,
# whereas Ninja normalizes '..'. Mixing caller-relative and absolute spellings
# can therefore create duplicate outputs before compilation even starts.
include_guard(GLOBAL)

# Caller source directory owns relative paths. Preserve the file's content hash
# and the caller's ordering/labels; normalization changes dependency identity only.
function(llaminar_native_source_fingerprint source_path output_digest)
    get_filename_component(_native_source_path "${source_path}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_native_source_path}")
    file(SHA256 "${_native_source_path}" _native_source_digest)
    set(${output_digest} "${_native_source_digest}" PARENT_SCOPE)
endfunction()
