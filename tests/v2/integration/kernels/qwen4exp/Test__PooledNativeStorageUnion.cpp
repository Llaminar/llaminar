/**
 * @file Test__PooledNativeStorageUnion.cpp
 * @brief Original MPI-owned pooled ngram storage-union capture and retirement.
 *
 * This owner uses the canonical integration MPI main: discovery and native
 * workers start inside the original process session, and all participants
 * retire before MPI finalization. Storage-only GPUs join the existing native
 * transport without becoming model readers. The shared provider fixture keeps
 * the same real row bytes, host service, graph stages and numerical oracle.
 */
#include "fixtures/qwen4exp/PooledImmutableNativeReplayBytes.h"
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "planning/ClusterInventoryGatherer.h"
#include "fixtures/qwen4exp/ImmutableLookupWorkerGateReplayAccess.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLEBuilder.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "memory/CPUMappedPublicationRoot.h"
#include "execution/qwen4exp/PooledImmutableLocalLookupGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantReadGraph.h"
#include "execution/qwen4exp/PooledImmutableCompleteCohort.h"
#include "execution/qwen4exp/PooledImmutableParticipantRetirementGraph.h"
#include "PooledACKReadyCancellation.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "tensors/NativeTensorExtent.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/OriginalRankLocalNativeReplay.h"
#include "fixtures/qwen4exp/PooledImmutableFactoryReplayAccess.h"
#include "execution/qwen4exp/PooledImmutableDirectedLookupGraph.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "backends/NativeOrderingEventGroup.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "transfer/TransferEngine.h"
#include "utils/EmbeddingVerifierFormats.h"
#include "utils/PooledNgramReplayTrace.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "kernels/qwen4exp/PooledImmutableNativeOps.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <type_traits>
#include <unistd.h>

namespace llaminar2::test {
namespace {
namespace q=qwen4exp;
namespace n=q::pooled_native;
namespace p=n::p;
#include "PooledImmutableProviderReplay.inc"
#include "PooledNativeStorageUnionReplay.inc"
}
}
