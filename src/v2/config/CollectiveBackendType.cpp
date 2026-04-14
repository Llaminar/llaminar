/**
 * @file CollectiveBackendType.cpp
 * @brief Implementation of CollectiveBackendType conversion functions
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "CollectiveBackendType.h"
#include <algorithm>
#include <cctype>

namespace llaminar2
{

    namespace
    {
        std::string toLower(const std::string &str)
        {
            std::string lower = str;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            return lower;
        }
    } // anonymous namespace

    const char *collectiveBackendTypeToString(CollectiveBackendType type)
    {
        switch (type)
        {
        case CollectiveBackendType::AUTO:
            return "auto";
        case CollectiveBackendType::NCCL:
            return "nccl";
        case CollectiveBackendType::RCCL:
            return "rccl";
        case CollectiveBackendType::HETEROGENEOUS:
            return "heterogeneous";
        case CollectiveBackendType::UPI:
            return "upi";
        case CollectiveBackendType::MPI:
            return "mpi";
        case CollectiveBackendType::HOST:
            return "host";
        default:
            return "unknown";
        }
    }

    std::optional<CollectiveBackendType> parseCollectiveBackendType(const std::string &str)
    {
        std::string lower = toLower(str);
        if (lower == "auto")
            return CollectiveBackendType::AUTO;
        if (lower == "nccl")
            return CollectiveBackendType::NCCL;
        if (lower == "rccl")
            return CollectiveBackendType::RCCL;
        if (lower == "heterogeneous" || lower == "hetero")
            return CollectiveBackendType::HETEROGENEOUS;
        if (lower == "upi")
            return CollectiveBackendType::UPI;
        if (lower == "mpi")
            return CollectiveBackendType::MPI;
        if (lower == "host")
            return CollectiveBackendType::HOST;
        return std::nullopt;
    }

} // namespace llaminar2
