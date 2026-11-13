/**
 * @file CPURouterKernelTestAdmission.h
 * @brief Exact explicit publication root for direct CPU kernel regressions.
 *
 * Direct kernel tests do not own a graph workspace manager. They author the
 * same immutable native declaration and PMA used by production router owners;
 * the final aligned backing retains its own claim across reset and replay.
 */
#pragma once
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "memory/CPURouterQ8StorageLayout.h"

namespace llaminar2::test
{
/**
 * @brief Bind one independently admitted final publication root before replay.
 * @param kernel Actual CPU implementation exercised by the fixture.
 * @param rows Maximum simultaneously published rows.
 * @param width Hidden columns, including any partial final native block.
 * @param allocator Exact CPU allocator identity under test.
 */
inline void admitCPURouterKernelForTest(CPUMoEKernel &kernel,int rows,int width,
    DeviceId allocator=DeviceId::cpu())
{
    const auto declaration=CPURouterQ8StorageLayout::declare(rows,width,allocator);
    PhysicalMemoryPlanBuilder plan;
    plan.add({0,allocator,declaration.physicalBytes(),declaration.physicalBytes()},
        PhysicalMemoryOwner::ExecutionWorkspace,declaration.physicalBytes());
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
    kernel.bindRouterQ8HiddenStorage(declaration.root.allocate(memory));
}
}
