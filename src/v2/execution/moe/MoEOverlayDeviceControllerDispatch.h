/**
 * @file MoEOverlayDeviceControllerDispatch.h
 * @brief Bind an immutable captured controller action to its native specialization.
 *
 * This is capture-time dispatch, not host execution policy. The device still
 * owns every transition, predicate, epoch and result. Compiling each already-
 * known action independently keeps unrelated policy branches out of its
 * register allocation and avoids the monolithic controller's register spills.
 */
#pragma once
#include "MoEOverlayDeviceControllerKernels.h"

namespace llaminar2
{
/**
 * @brief Visit the exact action as a compile-time value, or reject an invalid ABI value.
 * @param action Immutable action recorded into one graph node.
 * @param launch Generic callable with a boolean operator()<Action>().
 * @return The launcher's result, or false without invoking it for invalid actions.
 */
template <class Launch>
bool dispatchMoEOverlayControllerAction(
    MoEOverlayDeviceControllerAction action, Launch&& launch)
{
    using Action = MoEOverlayDeviceControllerAction;
    switch (action)
    {
    case Action::BeginTransaction:
        return launch.template operator()<Action::BeginTransaction>();
    case Action::PublishGroupSnapshot:
        return launch.template operator()<Action::PublishGroupSnapshot>();
    case Action::PublishCommand:
        return launch.template operator()<Action::PublishCommand>();
    case Action::AcknowledgePrepared:
        return launch.template operator()<Action::AcknowledgePrepared>();
    case Action::BeginCommit:
        return launch.template operator()<Action::BeginCommit>();
    case Action::AcknowledgePublished:
        return launch.template operator()<Action::AcknowledgePublished>();
    case Action::PublishAdmission:
        return launch.template operator()<Action::PublishAdmission>();
    case Action::BeginDynamicRetirement:
        return launch.template operator()<Action::BeginDynamicRetirement>();
    case Action::AcknowledgeRetired:
        return launch.template operator()<Action::AcknowledgeRetired>();
    case Action::CompleteDynamicRetirement:
        return launch.template operator()<Action::CompleteDynamicRetirement>();
    case Action::BeginLLEPRestore:
        return launch.template operator()<Action::BeginLLEPRestore>();
    case Action::AcknowledgeLLEPRestored:
        return launch.template operator()<Action::AcknowledgeLLEPRestored>();
    case Action::CompleteLLEPRestore:
        return launch.template operator()<Action::CompleteLLEPRestore>();
    case Action::AuthorStaticPolicy:
        return launch.template operator()<Action::AuthorStaticPolicy>();
    case Action::AwaitTransactionComplete:
        return launch.template operator()<Action::AwaitTransactionComplete>();
    case Action::AuthorDynamicPolicy:
        return launch.template operator()<Action::AuthorDynamicPolicy>();
    case Action::PublishParticipantSnapshot:
        return launch.template operator()<Action::PublishParticipantSnapshot>();
    case Action::ApplyRuntimeCandidate:
        return launch.template operator()<Action::ApplyRuntimeCandidate>();
    case Action::PublishRuntimeCandidate:
        return launch.template operator()<Action::PublishRuntimeCandidate>();
    case Action::PublishRuntimeRetirement:
        return launch.template operator()<Action::PublishRuntimeRetirement>();
    case Action::AwaitRuntimeCommit:
        return launch.template operator()<Action::AwaitRuntimeCommit>();
    case Action::AwaitRuntimeRetirement:
        return launch.template operator()<Action::AwaitRuntimeRetirement>();
    case Action::PublishRuntimeRetirementReadiness:
        return launch.template operator()<Action::PublishRuntimeRetirementReadiness>();
    case Action::CompleteEmptyDynamicDecision:
        return launch.template operator()<Action::CompleteEmptyDynamicDecision>();
    case Action::AuthorPreparedContextRestore:
        return launch.template operator()<Action::AuthorPreparedContextRestore>();
    case Action::Invalid:
        break;
    }
    return false;
}
} // namespace llaminar2
