/**
 * @file PagedKVSnapshotRef.h
 * @brief Backend-neutral typed paged snapshot reference (R21 J3 + requirements).
 *
 * The pooled resolver (owner) issues an exact reference to one published
 * snapshot: THE IDENTITY of the owning pooled authority (an authenticated id),
 * the published VERSION (never a reusable bank index), and the request slot
 * that referenced it. CPU/CUDA/ROCm share this one typed seam; a concrete
 * backend owns the actual pages and resolves the reference under
 * authentication (a bank may be reused and a version superseded — neither can
 * be authenticated by position).
 *
 * This type NEVER exposes a page pointer or a raw bank index: it is the
 * identity the graph consumes, and the pooled table resolves it to an
 * operation-readable view (or refuses) per execute.
 */
#pragma once
#include <cstdint>
#include <cstddef>

namespace llaminar2
{
/** @brief Exact typed reference to one pooled publication (identity + version). */
struct PagedKVSnapshotRef
{
    std::uint64_t authority_id = 0; ///< Authenticated pooled authority identity.
    std::uint64_t publication_version = 0; ///< Monotonic published version (never a bank index).
    std::uint64_t request_id = 0;     ///< Request identity that owns the slot mapping.
    int slot = -1;                    ///< Request slot (identity metadata, never a bank).
    bool valid() const noexcept { return authority_id != 0 && publication_version != 0; }
};

/**
 * @brief Operation-readable native paged view resolved from a ref.
 *
 * The concrete backend resolves a PagedKVSnapshotRef into one of these views
 * and retains it for the call. The native handle is backend-specific and
 * opaque to the shared interface: the CPU backend constructs its paged KV
 * view; CUDA/ROCm build their own. Page bytes are described by the backend
 * handle, never as raw host pointers here (no host shadow).
 */
struct PagedKVReadView
{
    const void *native_handle = nullptr; ///< Backend-resolved opaque handle.
    int rows = 0;        ///< Live rows in this publication.
    int heads = 0;       ///< KV heads.
    int dimension = 0;   ///< Head dimension.
    int page_tokens = 0; ///< Native block size.
    bool valid() const noexcept { return native_handle != nullptr && rows > 0; }
};
} // namespace llaminar2
