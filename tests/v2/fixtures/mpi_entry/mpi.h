/** @file mpi.h
 * @brief Inert MPI calls for the production entrypoint ordering replay.
 */
#pragma once
constexpr int MPI_THREAD_MULTIPLE = 3;
constexpr int MPI_THREAD_SINGLE = 0;
constexpr int MPI_SUCCESS = 0;
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
int MPI_Finalize();

constexpr int MPI_INT = 1, MPI_MIN = 2, MPI_COMM_WORLD = 3;
inline void *const MPI_IN_PLACE = reinterpret_cast<void *>(1);
int MPI_Initialized(int *value);
int MPI_Finalized(int *value);
int MPI_Allreduce(const void *send, void *receive, int count, int datatype, int op, int comm);
