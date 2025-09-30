#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

int main(int argc, char **argv)
{
    fprintf(stderr, "[minimal-mpi-probe] entering main before MPI_Init\n");
    fflush(stderr);
    int provided = 0;
    int required = MPI_THREAD_SERIALIZED;
    int rc = MPI_Init_thread(&argc, &argv, required, &provided);
    fprintf(stderr, "[minimal-mpi-probe] after MPI_Init rc=%d provided=%d\n", rc, provided);
    fflush(stderr);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    fprintf(stderr, "[minimal-mpi-probe] rank=%d size=%d\n", rank, size);
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    fprintf(stderr, "[minimal-mpi-probe] finalized\n");
    fflush(stderr);
    return 0;
}
