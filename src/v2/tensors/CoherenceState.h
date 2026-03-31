#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace llaminar2
{

// ============================================================================
// TensorCoherenceState — Explicit state machine for tensor host/device coherence
// ============================================================================
//
// Replaces the implicit boolean combination of (host_valid_, device_valid_,
// authoritative_device_, is_mapped_, is_bar_backed_) with a single enum that
// captures the VALID states a tensor can be in.
//
// State diagram:
//
//   ┌──────────┐  UPLOAD   ┌──────────┐  MARK_DEVICE_DIRTY  ┌─────────────────────┐
//   │ HOST_ONLY├──────────►│  SYNCED  ├─────────────────────►│DEVICE_AUTHORITATIVE │
//   └──────────┘           └────┬─────┘                      └──────────┬──────────┘
//        ▲                      │                                       │
//        │            MUTABLE   │                             DOWNLOAD  │
//        │          HOST_ACCESS │                                       │
//        │                      ▼                                       │
//        │              ┌──────────────────┐                            │
//        │              │HOST_AUTHORITATIVE│◄───────────────────────────┘
//        │              └────────┬─────────┘     (implicit download)
//        │                       │
//        │  RELEASE_DEVICE       │ UPLOAD
//        └───────────────────────┘
//
//   MAPPED tensors: Always valid on both sides. Transitions are no-ops.
//

enum class TensorCoherenceState : uint8_t
{
    HOST_ONLY,              // Data exists only on host. No GPU buffer allocated.
    HOST_AUTHORITATIVE,     // Both exist, host was modified more recently.
    DEVICE_AUTHORITATIVE,   // Both exist, device was modified more recently.
    SYNCED,                 // Both exist and are identical.
    MAPPED,                 // Zero-copy mapped memory. Both always valid.
    INVALID,                // Error state — should never be reached.
};

// Operations that trigger state transitions.
enum class CoherenceOp : uint8_t
{
    UPLOAD,              // Host → Device transfer (ensureOnDevice)
    DOWNLOAD,            // Device → Host transfer (ensureOnHost)
    MARK_DEVICE_DIRTY,   // GPU kernel wrote to device buffer
    MUTABLE_HOST_ACCESS, // CPU code obtained mutable host pointer
    RELEASE_DEVICE,      // GPU buffer freed
};

// Memory residency type — affects HOW transfers happen, not WHETHER.
enum class MemoryResidency : uint8_t
{
    STANDARD,   // Normal host + GPU allocations
    BAR_BACKED, // PCIe BAR cross-vendor memory (3-pointer: staging, BAR mmap, CUDA ptr)
    MAPPED,     // Zero-copy mapped (host-allocated, device-visible)
};

// ============================================================================
// Transition result
// ============================================================================

struct CoherenceTransition
{
    TensorCoherenceState new_state;
    bool valid;
};

// ============================================================================
// Transition table — constexpr, compile-time verifiable
// ============================================================================

// Number of states (excluding INVALID) and operations
inline constexpr int NUM_STATES = 5;
inline constexpr int NUM_OPS = 5;

// Transition table: [state][op] → {new_state, valid}
// MAPPED state is special: all transitions are valid and result in MAPPED.
inline constexpr std::array<std::array<CoherenceTransition, NUM_OPS>, NUM_STATES>
    COHERENCE_TRANSITIONS = {{
        // HOST_ONLY
        {{
            {TensorCoherenceState::SYNCED, true},              // UPLOAD
            {TensorCoherenceState::HOST_ONLY, true},           // DOWNLOAD (no-op, nothing on device)
            {TensorCoherenceState::INVALID, false},             // MARK_DEVICE_DIRTY (no device buffer!)
            {TensorCoherenceState::HOST_ONLY, true},           // MUTABLE_HOST_ACCESS (stays host-only)
            {TensorCoherenceState::HOST_ONLY, true},           // RELEASE_DEVICE (no-op)
        }},
        // HOST_AUTHORITATIVE
        {{
            {TensorCoherenceState::SYNCED, true},              // UPLOAD
            {TensorCoherenceState::HOST_AUTHORITATIVE, true},  // DOWNLOAD (no-op, host already valid)
            {TensorCoherenceState::DEVICE_AUTHORITATIVE, true}, // MARK_DEVICE_DIRTY
            {TensorCoherenceState::HOST_AUTHORITATIVE, true},  // MUTABLE_HOST_ACCESS (stays)
            {TensorCoherenceState::HOST_ONLY, true},           // RELEASE_DEVICE
        }},
        // DEVICE_AUTHORITATIVE
        {{
            {TensorCoherenceState::SYNCED, true},              // UPLOAD (device→host sync first, then upload is idempotent)
            {TensorCoherenceState::SYNCED, true},              // DOWNLOAD
            {TensorCoherenceState::DEVICE_AUTHORITATIVE, true}, // MARK_DEVICE_DIRTY (stays)
            {TensorCoherenceState::SYNCED, true},              // MUTABLE_HOST_ACCESS (implicit download)
            {TensorCoherenceState::HOST_ONLY, true},           // RELEASE_DEVICE (must download first!)
        }},
        // SYNCED
        {{
            {TensorCoherenceState::SYNCED, true},              // UPLOAD (no-op, already synced)
            {TensorCoherenceState::SYNCED, true},              // DOWNLOAD (no-op, already synced)
            {TensorCoherenceState::DEVICE_AUTHORITATIVE, true}, // MARK_DEVICE_DIRTY
            {TensorCoherenceState::HOST_AUTHORITATIVE, true},  // MUTABLE_HOST_ACCESS
            {TensorCoherenceState::HOST_ONLY, true},           // RELEASE_DEVICE
        }},
        // MAPPED
        {{
            {TensorCoherenceState::MAPPED, true},              // UPLOAD (no-op for mapped)
            {TensorCoherenceState::MAPPED, true},              // DOWNLOAD (no-op for mapped)
            {TensorCoherenceState::MAPPED, true},              // MARK_DEVICE_DIRTY
            {TensorCoherenceState::MAPPED, true},              // MUTABLE_HOST_ACCESS
            {TensorCoherenceState::MAPPED, true},              // RELEASE_DEVICE
        }},
    }};

// ============================================================================
// Pure functions — no side effects
// ============================================================================

/// Look up the transition for a given state and operation.
/// Returns {new_state, valid}. If valid is false, the transition is illegal.
inline constexpr CoherenceTransition coherenceTransition(TensorCoherenceState state, CoherenceOp op)
{
    auto state_idx = static_cast<int>(state);
    auto op_idx = static_cast<int>(op);

    if (state_idx < 0 || state_idx >= NUM_STATES || op_idx < 0 || op_idx >= NUM_OPS)
        return {TensorCoherenceState::INVALID, false};

    return COHERENCE_TRANSITIONS[state_idx][op_idx];
}

/// Check if a transition would be valid without applying it.
inline constexpr bool isValidTransition(TensorCoherenceState state, CoherenceOp op)
{
    return coherenceTransition(state, op).valid;
}

/// Apply a transition, returning the new state. Returns INVALID if illegal.
inline constexpr TensorCoherenceState applyTransition(TensorCoherenceState state, CoherenceOp op)
{
    auto result = coherenceTransition(state, op);
    return result.valid ? result.new_state : TensorCoherenceState::INVALID;
}

/// Does the current state require a device→host sync before host access?
inline constexpr bool needsDeviceToHostSync(TensorCoherenceState state)
{
    return state == TensorCoherenceState::DEVICE_AUTHORITATIVE;
}

/// Does the current state require a host→device upload?
inline constexpr bool needsHostToDeviceUpload(TensorCoherenceState state)
{
    return state == TensorCoherenceState::HOST_ONLY || state == TensorCoherenceState::HOST_AUTHORITATIVE;
}

/// Is host data currently valid (safe to read without sync)?
inline constexpr bool isHostValid(TensorCoherenceState state)
{
    switch (state)
    {
    case TensorCoherenceState::HOST_ONLY:
    case TensorCoherenceState::HOST_AUTHORITATIVE:
    case TensorCoherenceState::SYNCED:
    case TensorCoherenceState::MAPPED:
        return true;
    case TensorCoherenceState::DEVICE_AUTHORITATIVE:
    case TensorCoherenceState::INVALID:
        return false;
    }
    return false;
}

/// Is device data currently valid (safe for GPU kernels)?
inline constexpr bool isDeviceValid(TensorCoherenceState state)
{
    switch (state)
    {
    case TensorCoherenceState::DEVICE_AUTHORITATIVE:
    case TensorCoherenceState::SYNCED:
    case TensorCoherenceState::MAPPED:
        return true;
    case TensorCoherenceState::HOST_ONLY:
    case TensorCoherenceState::HOST_AUTHORITATIVE:
    case TensorCoherenceState::INVALID:
        return false;
    }
    return false;
}

// ============================================================================
// String conversions
// ============================================================================

inline constexpr std::string_view to_string(TensorCoherenceState state)
{
    switch (state)
    {
    case TensorCoherenceState::HOST_ONLY:
        return "HOST_ONLY";
    case TensorCoherenceState::HOST_AUTHORITATIVE:
        return "HOST_AUTHORITATIVE";
    case TensorCoherenceState::DEVICE_AUTHORITATIVE:
        return "DEVICE_AUTHORITATIVE";
    case TensorCoherenceState::SYNCED:
        return "SYNCED";
    case TensorCoherenceState::MAPPED:
        return "MAPPED";
    case TensorCoherenceState::INVALID:
        return "INVALID";
    }
    return "UNKNOWN";
}

inline constexpr std::string_view to_string(CoherenceOp op)
{
    switch (op)
    {
    case CoherenceOp::UPLOAD:
        return "UPLOAD";
    case CoherenceOp::DOWNLOAD:
        return "DOWNLOAD";
    case CoherenceOp::MARK_DEVICE_DIRTY:
        return "MARK_DEVICE_DIRTY";
    case CoherenceOp::MUTABLE_HOST_ACCESS:
        return "MUTABLE_HOST_ACCESS";
    case CoherenceOp::RELEASE_DEVICE:
        return "RELEASE_DEVICE";
    }
    return "UNKNOWN";
}

inline constexpr std::string_view to_string(MemoryResidency residency)
{
    switch (residency)
    {
    case MemoryResidency::STANDARD:
        return "STANDARD";
    case MemoryResidency::BAR_BACKED:
        return "BAR_BACKED";
    case MemoryResidency::MAPPED:
        return "MAPPED";
    }
    return "UNKNOWN";
}

} // namespace llaminar2
