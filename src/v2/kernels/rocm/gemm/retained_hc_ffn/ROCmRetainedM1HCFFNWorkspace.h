/**
 * @file ROCmRetainedM1HCFFNWorkspace.h
 * @brief Shared host declaration behind graph workspace materialization and PMA.
 */
#pragma once

#include "execution/moe/MoEWorkspaceRequirements.h"
#include "ROCmRetainedM1HCFFNDevice.h"

namespace llaminar2::rocm {
/**
 * @brief Declare the retained model's existing serial-graph-family scratch.
 * @return The exact named buffer contribution consumed by graph materialization and PMA.
 */
inline WorkspaceRequirements retainedM1HCFFNWorkspace()
{
    WorkspaceRequirements requirements;
    MoEWorkspaceBuffers::add(requirements,MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,
        fullNativeHCFFNSlotBytes(FullNativeHCFFNImplementation::RetainedGFX906M1)*
            MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,WorkspaceContentLifetime::SerialGraphFamily);
    return requirements;
}
}
