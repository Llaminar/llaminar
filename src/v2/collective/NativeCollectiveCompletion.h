/** @file NativeCollectiveCompletion.h
 * @brief Original native byte receive evidence for exact dependent device ACK.
 * A view borrows persistent admitted metadata and immutable communicator identity;
 * it creates no allocation, scheduler state, communicator or alternate ledger.
 */
#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
/** @brief Original device operation disposition; zero is deliberately uncompleted. */
enum class ncclLlaminarCompletionPhase : uint32_t { Pending = 0, Completed = 1, Aborted = 2 };
/**
 * @brief Immutable layout of a terminal-only original connector observation.
 * Every reported channel is an actual native channel. Its two step words are
 * observations, never a second progress ledger or an inference decision input.
 */
struct ncclLlaminarP2pProgressShape {
  uint64_t bytes;
  uint32_t channels;
  uint32_t reserved;
};
/** @brief Sum followed by one sent/received UINT64 pair per original channel. */
struct ncclLlaminarP2pProgress {
  uint64_t sentSteps;
  uint64_t receivedSteps;
};
static_assert(sizeof(ncclLlaminarP2pProgressShape)==16,"Native progress layout ABI");
static_assert(sizeof(ncclLlaminarP2pProgress)==16,"Native progress observation ABI");

/** @brief Borrowed immutable provenance of one original native communicator. */
struct ncclLlaminarCompletionAuthority {
  const volatile uint32_t* abortFlag;
  const void* deviceCommunicator;
};
/** @brief Native-written receive evidence, observed only after the whole original launch. */
struct ncclLlaminarByteCompletion {
  uint64_t table, admission, sequence, bytes;
  const volatile uint32_t* nativeAbortFlag;
  const void* nativeCommunicator;
  uint64_t message;
  ncclLlaminarCompletionPhase phase;
  uint32_t reserved;
};
/**
 * @brief Borrow original immutable-for-operation frame fields and sticky error authority.
 * Native work never advances these fields or clears an error. The receipt and
 * passive counter are protocol outputs, not execution/scheduling authorities.
 */
struct ncclLlaminarCompletionOperand {
  const uint64_t* table;
  const uint64_t* admission;
  const uint64_t* sequence;
  int32_t* firstError;
  ncclLlaminarByteCompletion* receipt;
  unsigned long long* outgoingPayloadBytes;
  uint64_t message;
  int32_t failureCode;
  uint32_t reserved;
};
static_assert(sizeof(ncclLlaminarByteCompletion) == 64 && sizeof(ncclLlaminarCompletionOperand) == 64 && sizeof(ncclLlaminarCompletionAuthority) == 16,
    "Native completion uses fixed protocol metadata, never capacity-sized storage");

/**
 * @brief Exact completion law shared by device ACK and recorded actual-body replay.
 * @param receipt Actual native output after its whole-launch dependency.
 * @param table Original source table identity.
 * @param admission Original retained interval admission.
 * @param sequence Original lookup sequence.
 * @param bytes Exact expected live bytes, including zero.
 * @param message Original static operation key.
 * @param expected Original communicator capability, not receipt-selected pointers.
 * @param nativeAbort Acquired original flag after the whole native launch dependency.
 * @param firstError Snapshot of the same original sticky failure field.
 * @return True only for fresh successful receive evidence and exact original identity.
 * This observes neither a byte counter nor host execution state. A prior error
 * vetoes success even when a different channel published Completed first.
 */
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
constexpr bool ncclLlaminarCompletionMatches(const ncclLlaminarByteCompletion& receipt,
    uint64_t table, uint64_t admission, uint64_t sequence, uint64_t bytes,
    uint64_t message, ncclLlaminarCompletionAuthority expected, uint32_t nativeAbort, int32_t firstError) {
  return table && admission && sequence && message && expected.abortFlag && expected.deviceCommunicator && !nativeAbort && !firstError &&
      receipt.reserved == 0 && receipt.phase == ncclLlaminarCompletionPhase::Completed &&
      receipt.table == table && receipt.admission == admission && receipt.sequence == sequence &&
      receipt.bytes == bytes && receipt.message == message && receipt.nativeAbortFlag == expected.abortFlag &&
      receipt.nativeCommunicator == expected.deviceCommunicator;
}


#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_NATIVE_COMPLETION_HD __host__ __device__
#else
#define LLAMINAR_NATIVE_COMPLETION_HD
#endif
namespace llaminar2 {
namespace test { class PooledNativeCollectiveReplayAccess; }
class NativeReceiveCompletionView;
/** @brief Immutable borrowed identity minted by the original native coordinator only. */
class NativeCompletionAuthority final {
public:
    /** @brief Explicitly unbound default for unsupported/native-unprepared contexts. */
    NativeCompletionAuthority() = default;
    /** @return Whether both original native identities are present. */
    [[nodiscard]] bool valid() const noexcept { return original_.abortFlag && original_.deviceCommunicator; }
    /** @return Original immutable pointer pair; no GPU value is read. */
    [[nodiscard]] ncclLlaminarCompletionAuthority binding() const noexcept { return original_; }
    /** @brief Project exact admitted receipt storage after caller's original arena binding.
     * @param receipt Persistent actual-native output root; source capture authenticates its PMA/offset.
     * @return Read-only projection retaining this original communicator identity.
     * @throws std::invalid_argument For absent authority or misaligned metadata.
     */
    [[nodiscard]] NativeReceiveCompletionView receive(const ncclLlaminarByteCompletion* receipt) const;
private:
    // Construction-only replay grants no private receive-view access or completion evidence.
    friend class test::PooledNativeCollectiveReplayAccess;
    friend class NCCLCoordinator;
    friend class RCCLCoordinator;
    /** @brief Import original coordinator metadata; the closed replay friend is construction-only. */
    explicit NativeCompletionAuthority(ncclLlaminarCompletionAuthority original) : original_(original) {}
    ncclLlaminarCompletionAuthority original_{};
};
/** @brief Device-only receive observation; ACK must depend on the whole original native launch. */
class NativeReceiveCompletionView final {
public:
    /** @return Explicit absence for a mapped receipt; never accepted by native ACK. */
    [[nodiscard]] LLAMINAR_NATIVE_COMPLETION_HD static constexpr NativeReceiveCompletionView notRequired() noexcept
    { return NativeReceiveCompletionView(); }
    /** @return Whether this view retains its original native communicator and actual receipt. */
    [[nodiscard]] LLAMINAR_NATIVE_COMPLETION_HD constexpr bool valid() const noexcept
    { return original_.abortFlag && original_.deviceCommunicator && receipt_; }
    /** @return Actual original abort address; acquire it on device before calling matches. */
    [[nodiscard]] LLAMINAR_NATIVE_COMPLETION_HD constexpr const volatile std::uint32_t* abortAddress() const noexcept { return original_.abortFlag; }
    /** @return Actual original receipt; never dereference this GPU pointer on host. */
    [[nodiscard]] LLAMINAR_NATIVE_COMPLETION_HD constexpr const ncclLlaminarByteCompletion* receipt() const noexcept { return receipt_; }
    /** @return Immutable original native communicator identity for capture sealing. */
    [[nodiscard]] LLAMINAR_NATIVE_COMPLETION_HD constexpr ncclLlaminarCompletionAuthority authority() const noexcept { return original_; }
    /** @brief Device-only exact receipt check after the whole native launch dependency.
     * @param table Actual original table identity from the received frame.
     * @param admission Actual original retained interval admission.
     * @param sequence Actual original lookup sequence, including repeated/reset trajectories.
     * @param bytes Exact live received bytes, including a valid zero.
     * @param message Original static operation key, never inferred from bytes.
     * @param nativeAbort Acquired value from abortAddress(), not a receipt-selected flag.
     * @param firstError Original sticky first-error snapshot; never cleared by communication.
     * @return Whether this actual original native receive can authorize existing ACK checks.
     */
    [[nodiscard]] LLAMINAR_NATIVE_COMPLETION_HD bool matches(std::uint64_t table, std::uint64_t admission,
        std::uint64_t sequence, std::uint64_t bytes, std::uint64_t message,
        std::uint32_t nativeAbort, std::int32_t firstError) const noexcept {
        if (!valid()) return false;
        return ncclLlaminarCompletionMatches(*receipt_, table, admission, sequence, bytes,
            message, original_, nativeAbort, firstError);
    }
private:
    friend class NativeCompletionAuthority;
    /** @brief Explicit unbound representation, available only through notRequired(). */
    LLAMINAR_NATIVE_COMPLETION_HD constexpr NativeReceiveCompletionView() noexcept = default;
    /** @brief Closed metadata projection; owns no live value and cannot reset/rearm native state. */
    NativeReceiveCompletionView(ncclLlaminarCompletionAuthority original, const ncclLlaminarByteCompletion* receipt)
        : original_(original), receipt_(receipt) {}
    ncclLlaminarCompletionAuthority original_{};
    const ncclLlaminarByteCompletion* receipt_ = nullptr;
};
inline NativeReceiveCompletionView NativeCompletionAuthority::receive(const ncclLlaminarByteCompletion* receipt) const {
    if (!valid() || !receipt || reinterpret_cast<std::uintptr_t>(receipt) % alignof(ncclLlaminarByteCompletion))
        throw std::invalid_argument("Native completion view requires original communicator and admitted aligned receipt");
    return NativeReceiveCompletionView(original_, receipt);
}
}

#undef LLAMINAR_NATIVE_COMPLETION_HD
