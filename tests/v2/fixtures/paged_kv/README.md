# Private page-authority recording v1

`private_ownership_v1.txt` records actual PMA-issued CPU pool inputs and observed
results, captured by `PhysicalMemoryAuthority.PagePoolRecordedUnpublishedOwnershipReplay`.
The initial pool is explicitly empty with zero frame/request generations. A new
process binds its own pool incarnation to the trace's single pool namespace;
frame indices, frame generations and every accept/refuse result are compared
exactly. No KV payload or native access readiness exists in this trace.

Capture evidence: `CB-checkpoints/paged-p1-recording.rcAtQ6`, on implementation
base `a0b69f633` plus the preserved uncommitted P1 prototype. This is a small
controller fixture, not a model corpus or a qualification receipt. The recording
is not rewritten by normal Unit or preflight runs. Diagnostic recording requires
explicit `LLAMINAR_RECORD_PRIVATE_PAGE_TRACE` and a destination path; changing
expected results requires inspection and independent review, never automatic
rebaselining of a failed replay.

The sequence holds one private-frame pin beyond request retirement, observes
continued admission pressure, releases the final pin, then observes the same
frame under a newer incarnation. It preserves a peer throughout. `private_append_v1.txt` extends the recorded sequence with actual synchronous
CPU byte append plans and immutable byte publications. Capture evidence:
`CB-checkpoints/paged-p1-append-recording.1bKRNr`. It records refusal while a
completed publication is pinned, an aborted append whose written suffix never
becomes visible, correction at the original valid end under a non-reused content
version, and a byte publication surviving request retirement. The original
ownership-only trace remains unchanged.

`private_readers_v1.txt` adds actual registration, closure, request retirement,
pressure refusal and subsequent CPU consumption of the original bytes. Capture:
`CB-checkpoints/paged-p1-reader-recording.U0jSsn`. The first two recordings remain
unchanged. CMake embeds these small retained fixtures into the test binary, so
installed compiler/source-free runners do not depend on builder-local paths.

Logical KV-table/operation-readable views, sharing/COW, tentative verifier
causality, native access/transfer completion and GPU traces remain separate
future requirements. The raw-byte publication trace does not qualify a native
KV codec, append kernel or attention consumer.
