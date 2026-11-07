/**
 * @file ForwardGraphSubmissionIntent.h
 * @brief One immutable construction purpose shared by main and MTP forward graphs.
 *
 * Geometry declarations retain typed buffer requirements without becoming
 * executable bindings or allocating native ticket payloads.
 */
#pragma once
#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Submission lifecycle requested for one forward graph.
     *
     * Production inference normally builds and launches the graph atomically.
     * A setup authority may instead capture and instantiate the exact executable
     * while leaving all model arithmetic, manual boundaries, output publication,
     * and transaction numbering untouched. The first ordinary invocation then
     * owns transaction zero. This policy is deliberately carried beside the
     * input rather than inferred from missing request data.
     */
    enum class ForwardGraphSubmissionIntent : uint8_t
    {
        Execute = 0, ///< Submit one production forward transaction.
        MaterializeExecutableWithoutLaunch, ///< Setup-only capture/instantiation.
        WorkspaceDeclaration, ///< Geometry/lifetime manifest only; never executable or capturable.
    };
}
