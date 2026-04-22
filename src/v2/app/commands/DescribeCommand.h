/**
 * @file DescribeCommand.h
 * @brief 'llaminar describe' — print cluster/device inventory and exit.
 *
 * MPI-aware command that bootstraps MPI to gather the full cluster inventory
 * via MPI_Allgatherv, exercising the same code path as real inference.
 * Use --no-mpi-bootstrap for local-only mode.
 */

#pragma once

#include "app/ICommand.h"

namespace llaminar2
{

    class DescribeCommand : public ICommand
    {
    public:
        const char *name() const override { return "describe"; }
        const char *description() const override { return "Print cluster/device inventory and exit"; }
        int execute(int argc, char *argv[]) override;
    };

} // namespace llaminar2
