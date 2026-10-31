/** @file mpi.h
 * @brief Inert MPI calls for the production entrypoint ordering replay.
 */
#pragma once
constexpr int MPI_THREAD_MULTIPLE = 3;
constexpr int MPI_THREAD_SINGLE = 0;
constexpr int MPI_SUCCESS = 0;
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
int MPI_Finalize();
