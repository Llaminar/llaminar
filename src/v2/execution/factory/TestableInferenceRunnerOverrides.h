/**
 * @file TestableInferenceRunnerOverrides.h
 * @brief Explicit model-light factory admission at the existing test boundary.
 *
 * Mock model fixtures have no concrete model-owned allocation authority. They
 * must author one from the finalized graph declaration before materialization.
 * Concrete model contexts reject this override and retain their model issuer.
 */
#pragma once
#include <functional>
#include <memory>
namespace llaminar2
{
class IGraphBuilder;
class PhysicalMemoryAuthority;
/** @brief Explicit test-only dependencies, never a production admission policy. */
struct TestableInferenceRunnerOverrides
{
    /**
     * @brief Author exact fixture admission after final TP/PP graph resolution.
     * @param graph Final source-bound builder before any arena materialization.
     * @return Mandatory fixture-owned canonical authority; null is rejected.
     */
    std::function<std::shared_ptr<PhysicalMemoryAuthority>(
        const std::shared_ptr<IGraphBuilder>&)> test_graph_allocation_admission;
};
}
