/**
 * @file llaminar_native_storage_authority_v1.h
 * @brief Framework-free exact native-root preclaim boundary shared by both SDKs.
 *
 * The native allocation owner supplies allocator-derived physical geometry
 * before Create. PMA alone admits/reserves bytes and returns a unique opaque
 * token. That token stays on the actual native root through all borrowed aliases
 * and retires only after final physical destruction. There is no pointer ledger.
 */
#ifndef LLAMINAR_NATIVE_STORAGE_AUTHORITY_V1_H
#define LLAMINAR_NATIVE_STORAGE_AUTHORITY_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/** @brief Resource identity; host NUMA placement does not create another ledger. */
enum llaminarNativeResourceV1 {
  llaminarNativeCUDADeviceV1=1, llaminarNativeROCmDeviceV1=2, llaminarNativeHostCPU0V1=3
};
/** @brief Actual root family, never an inactive capability advertisement. */
enum llaminarNativeRootFamilyV1 {
  llaminarNativeCommMetadataV1=1, llaminarNativeSharedPeersV1=2,
  llaminarNativeWorkFifoV1=3, llaminarNativeProfilerV1=4,
  llaminarNativeAbortFlagV1=5, llaminarNativeP2pBackingV1=6,
  llaminarNativeShmBackingV1=7, llaminarNativeCapturedWorkV1=8,
  llaminarNativeProcessSDKV1=9, llaminarNativeDriverContextV1=10
};
/** @brief Existing native lifetime that physically owns the allocation. */
enum llaminarNativeRootLifetimeV1 {
  llaminarNativeCommLifetimeV1=1, llaminarNativeSharedLifetimeV1=2,
  llaminarNativeProxyLifetimeV1=3, llaminarNativeGraphLifetimeV1=4,
  llaminarNativeProcessLifetimeV1=5
};
/** @brief Explicit allocator semantics whose extent is known before Create. */
enum llaminarNativeAllocatorV1 {
  llaminarNativeCUDADeviceVmmV1=1, llaminarNativeCUDAHostVmmV1=2,
  llaminarNativeROCmDeviceVmmV1=3, llaminarNativeROCmUncachedDeviceVmmV1=4,
  llaminarNativeROCmUncachedHostVmmV1=5, llaminarNativeRegisteredHostFileV1=6,
  llaminarNativeHostAnonymousPagesV1=7
};
/** @brief Both initialization phases require claims before physical creation. */
enum llaminarNativeSetupPhaseV1 {
  llaminarNativeColdProbeV1=1, llaminarNativeRetainedSetupV1=2,
  llaminarNativeGraphRecordingHostMetadataV1=3
};
/** @brief Fixed status domain: no exceptions cross the dependency C ABI. */
enum llaminarNativeStorageStatusV1 {
  llaminarNativeStorageSuccessV1=0, llaminarNativeStorageInvalidV1=1,
  llaminarNativeStorageCapacityV1=2, llaminarNativeStorageUnsupportedV1=3,
  llaminarNativeStorageInternalV1=4
};

/**
 * @brief Exact creation request; all bytes describe one owned physical root.
 *
 * owner_identity names the stable native owner object, not its mapped address.
 * shared_anchor_identity names its existing refcount/shared lifetime owner (or
 * owner_identity for an exclusive root). Borrowed aliases never submit requests.
 * allocator_flags retains exact native allocation/handle-policy flags; requested
 * and physical extents plus granularity must match the actual Create call.
 */
struct llaminarNativeRootRequestV1 {
  uint64_t abi_version;
  uint64_t record_bytes;
  uint64_t owner_identity;
  uint64_t shared_anchor_identity;
  uint64_t requested_bytes;
  uint64_t physical_bytes;
  uint64_t allocation_granularity;
  uint64_t allocator_flags;
  uint64_t setup_epoch; /**< Must equal the immutable admission context epoch. */
  uint32_t resource_kind;
  int32_t resource_ordinal; /**< GPU ordinal, or exactly zero for CPU0. */
  int32_t host_numa; /**< Placement metadata: -1 explicit Host policy, >=0 HostNuma. */
  uint32_t owner_family;
  uint32_t lifetime;
  uint32_t allocator_kind;
  uint32_t phase;
  uint32_t reserved_zero;
};

/**
 * @brief Read-only prepared native-context facts issued by the real backend owner.
 *
 * The application authenticates runtime_generation against that backend before
 * admitted init; nonzero caller bits are not proof. The SDK verifies actual
 * current context/device/stack and binding epoch before allocating any root.
 * It never grows context storage, probes another limit or retries another path.
 */
struct llaminarNativePreparedContextV1 {
  uint64_t abi_version;
  uint64_t record_bytes;
  uint32_t backend_kind; /**< CUDADevice or ROCmDevice resource tag. */
  int32_t device_ordinal;
  uint64_t native_context_identity;
  uint64_t setup_epoch;
  uint64_t prepared_stack_limit;
  uint64_t runtime_generation;
  uint64_t reserved_zero;
};

/**
 * @brief Borrowed admission callbacks retained by the enclosing native session.
 *
 * claim atomically reserves through one canonical PMA state before Create and
 * initializes out_token only on success. The token owns its PMA lifetime and
 * release needs no expired initialization context. Failed Create cancels the
 * preclaim through release; a failed physical cleanup MUST retain the token.
 * materialize publishes successful physical owner creation before consumers;
 * publication failure cleans physical backing before releasing the preclaim.
 * release clears the actual root field only on success; a null field is rejected.
 * The application's native session outlives all setup/capture/SDK teardown.
 */
struct llaminarNativeStorageAuthorityV1 {
  uint64_t abi_version;
  uint64_t record_bytes;
  void* context;
  int32_t (*claim)(void* context, const struct llaminarNativeRootRequestV1* request, void** out_token);
  int32_t (*materialize)(void* token);
  int32_t (*release)(void** inout_token);
  uint64_t setup_epoch;
  uint64_t reserved_zero;
};

#ifdef __cplusplus
}
static_assert(sizeof(llaminarNativeRootRequestV1)==104, "Native root request ABI");
static_assert(sizeof(llaminarNativeStorageAuthorityV1)==64, "Native authority ABI");
static_assert(sizeof(llaminarNativePreparedContextV1)==64, "Native prepared context ABI");
#endif
#endif
