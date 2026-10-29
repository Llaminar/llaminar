/**
 * @file WorkspaceDescriptor.cpp
 * @brief Out-of-line typed CPU root metadata composition for workspace requirements.
 *
 * Keeping CPU layout construction here preserves lightweight ubiquitous GPU
 * workspace headers. This composes declarations only; PMA owns live claims.
 */
#include "WorkspaceDescriptor.h"
#include "memory/CPUGroupedMoEStorageLayout.h"
#include "memory/CPURouterQ8StorageLayout.h"
namespace llaminar2
{
size_t WorkspaceRequirements::cpu_grouped_bytes() const
{
    if (cpu_router_q8_storage)
        cpu_router_q8_storage->validate();
    if (cpu_grouped_moe_storage)
    {
        cpu_grouped_moe_storage->validate();
        if (cpu_router_q8_storage && !cpu_grouped_moe_storage->routerLayout().contains(*cpu_router_q8_storage))
            throw std::invalid_argument("CPU router publication exceeds its complete grouped workspace envelope");
        return cpu_grouped_moe_storage->physicalBytes();
    }
    return cpu_router_q8_storage ? cpu_router_q8_storage->physicalBytes() : 0;
}
void WorkspaceRequirements::merge(const WorkspaceRequirements &other)
{
    if (other.cpu_grouped_moe_storage)
    {
        other.cpu_grouped_moe_storage->validate();
        cpu_grouped_moe_storage = std::make_shared<const CPUGroupedMoEStorageLayout>(
            cpu_grouped_moe_storage
                ? cpu_grouped_moe_storage->merge(*other.cpu_grouped_moe_storage)
                : *other.cpu_grouped_moe_storage);
    }
    if (other.cpu_router_q8_storage)
    {
        other.cpu_router_q8_storage->validate();
        cpu_router_q8_storage = std::make_shared<const CPURouterQ8StorageLayout>(
            cpu_router_q8_storage
                ? cpu_router_q8_storage->merge(*other.cpu_router_q8_storage)
                : *other.cpu_router_q8_storage);
    }
    for (const auto &buf : other.buffers)
    {
        bool found = false;
        for (auto &existing : buffers)
        {
            if (existing.name == buf.name)
            {
                // Keep larger size
                if (buf.size_bytes > existing.size_bytes)
                {
                    existing.size_bytes = buf.size_bytes;
                }
                // Keep stricter alignment
                if (buf.alignment > existing.alignment)
                {
                    existing.alignment = buf.alignment;
                }
                // Keep required if either is required
                existing.required = existing.required || buf.required;
                // A name used in more than one regime is a common
                // participant buffer and must retain one stable address.
                if (existing.regime != buf.regime)
                {
                    existing.regime =
                        WorkspaceExecutionRegime::Any;
                }
                // Persistent content is the stronger contract. One
                // initialize-once user makes the shared name live for
                // the complete serial graph family.
                if (buf.content_lifetime ==
                    WorkspaceContentLifetime::SerialGraphFamily)
                {
                    existing.content_lifetime =
                        WorkspaceContentLifetime::SerialGraphFamily;
                }
                found = true;
                break;
            }
        }
        if (!found)
        {
            buffers.push_back(buf);
        }
    }
}
}
