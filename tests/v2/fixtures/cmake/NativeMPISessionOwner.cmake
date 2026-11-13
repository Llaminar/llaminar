# @file NativeMPISessionOwner.cmake
# @brief Import the real CPU-only process owner into language-free registration fixtures.
# Installed test runners need no compiler or substitute launch implementation.
if(NOT DEFINED ENV{LLAMINAR_TEST_MPI_SESSION_HELPER}
   OR NOT EXISTS "$ENV{LLAMINAR_TEST_MPI_SESSION_HELPER}")
    message(FATAL_ERROR "Registration fixture requires the build-owned MPI session helper")
endif()
add_executable(llaminar_mpi_session IMPORTED)
set_target_properties(llaminar_mpi_session PROPERTIES
    IMPORTED_LOCATION "$ENV{LLAMINAR_TEST_MPI_SESSION_HELPER}")
