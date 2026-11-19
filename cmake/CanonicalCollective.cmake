# Admit the artifact before emitting the loader's path. A relocated prefix is
# supported when it carries the source builder's proof or the shipped ELF.
set(LLAMINAR_PUBLISHED_COLLECTIVE_IMAGE "" CACHE STRING
    "Immutable published Llaminar runtime providing the selected collective libraries")

function(llaminar_verify_collective backend library report receipt)
    set(verifier "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../scripts/build/verify_collective_dependency.py")
    if(LLAMINAR_PUBLISHED_COLLECTIVE_IMAGE)
        set(provenance --published-image "${LLAMINAR_PUBLISHED_COLLECTIVE_IMAGE}")
    else()
        set(provenance --report "${report}" --receipt "${receipt}")
    endif()
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${verifier}" "${backend}"
        --library "${library}" --architectures "${RCCL_GPU_TARGETS}" ${provenance}
        COMMAND_ERROR_IS_FATAL ANY)
endfunction()
