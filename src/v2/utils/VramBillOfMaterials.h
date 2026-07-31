/**
 * @file VramBillOfMaterials.h
 * @brief Shared helpers for per-buffer VRAM allocation diagnostics.
 */

#pragma once

#include "DebugEnv.h"
#include "Logger.h"

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>

namespace llaminar2
{
    inline bool vramBomEnabled()
    {
        return debugEnv().vram_bom;
    }

    inline std::string vramBomMiB(size_t bytes)
    {
        std::ostringstream out;
        out << std::fixed << std::setprecision(3)
            << (static_cast<double>(bytes) / (1024.0 * 1024.0));
        return out.str();
    }

    inline std::string vramBomBytes(size_t bytes)
    {
        std::ostringstream out;
        out << "bytes=" << bytes << " mib=" << vramBomMiB(bytes);
        return out.str();
    }

    /**
     * @brief Render an allocation address in a stable, joinable BOM field.
     *
     * Backend allocation records and higher-level tensor ownership records use
     * the same representation so an offline diagnostic can attribute every
     * live raw allocation without maintaining an always-on runtime registry.
     */
    inline std::string vramBomPointer(const void *ptr)
    {
        std::ostringstream out;
        out << ptr;
        return out.str();
    }

    inline void logVramBomLine(const std::string &kind, const std::string &fields)
    {
        if (!vramBomEnabled())
            return;
        LOG_INFO("[VRAM_BOM] kind=" << kind << " " << fields);
    }
} // namespace llaminar2
