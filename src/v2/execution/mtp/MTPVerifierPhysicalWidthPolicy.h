/**
 * @file MTPVerifierPhysicalWidthPolicy.h
 * @brief Immutable verifier geometry shared by capture and learned economics.
 *
 * This scalar names the existing physical execution contract. It is not a
 * backend capability probe or a second graph-family registry.
 */
#pragma once
namespace llaminar2
{
    /** @brief Mapping from device-owned logical rows to retained graph width. */
    enum class MTPVerifierPhysicalWidthPolicy
    {
        BoundedLogicalBucket,
        DynamicDeviceEnvelope,
        ExactLogicalRows,
    };
}
