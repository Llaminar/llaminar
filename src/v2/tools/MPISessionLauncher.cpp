/**
 * @file MPISessionLauncher.cpp
 * @brief CPU-only MPI launcher frontend shared by production and CTest.
 *
 * All ownership, exact-process retirement and bounded directory cleanup live
 * in MPISessionOwner. This executable initializes no model, MPI or accelerator.
 */
#include "utils/MPISessionOwner.h"
int main(int argc, char **argv)
{
    return llaminar2::MPISessionOwner::launcherMain(argc, argv);
}
