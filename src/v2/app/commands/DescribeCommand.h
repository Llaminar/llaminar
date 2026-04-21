/**
 * @file DescribeCommand.h
 * @brief 'llaminar describe' — print cluster/device inventory and exit.
 *
 * Lightweight command that queries CPU topology, GPU devices, and NUMA
 * configuration without loading a model or bootstrapping MPI.
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
