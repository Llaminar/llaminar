# P0 §8.4 proposal: client retention value-to-class map, minimum lifetimes, default class, operator ceiling, unknown/cache-field handling — and baseline matrix freeze

**Status:** Proposal for owner decision (P0 §8.4; plan §4.3 "Retention lifetime classes" row).
**Scope:** No code changes. This documents the settled semantic contract proposed for implementation
in the prefix-cache/tier work (P3 addition to the existing resident tier; no new API surface).
**Source bindings:** §8.4 table (design `design_continuous_batching_20261105.md` lines 687–745),
§4.3 configuration contract (lines 228–245), §8.2 value score, §8.1 trust domains.

---

## 1. Value-to-class map (accepted standard request values)

Only standard fields of APIs Llaminar actually serves are interpreted. Today that is **Chat
Completions** (`prompt_cache_key`, `prompt_cache_retention`). Responses/Anthropic are served only if
and when they are; the map below is written so adding them later is a table extension, not a new policy.

| Served API | Standard field | Accepted values | Typed class | Minimum lifetime (request-relative) |
|---|---|---|---|---|
| Chat Completions | `prompt_cache_retention` | `in_memory` | `InMemory` (rank 0) | 1 request lifetime (the writing request's lifetime; effectively "as long as the serving process holds it under normal pressure") |
| Chat Completions | `prompt_cache_retention` | `24h` | `TieredDay` (rank 1) | 24 h, measured from the start of the request that wrote or last effectively used it (§8.2 refresh under the same receipt) |
| Chat Completions | `prompt_cache_retention` | absent | `Default` (rank 0.5, between `InMemory` and `TieredDay`) — **see §3** | Default lifetime (see §3) |
| Chat Completions | `prompt_cache_retention` | anything else | — | `invalid_request_error`; never silently mapped |

**Ordering:** `InMemory` < `Default` < `TieredDay` (ascending retention rank; victim order uses
unexpired class **longest first**, so `TieredDay` entries are protected before `Default`, and
`Default` before `InMemory`).

## 2. Class semantics (unchanged from §8.4, restated for the map)

- **Namespace is not trust.** `prompt_cache_key` partitions retention and accounting; it never makes
  incompatible content shareable or grants cross-trust-domain access (§8.1).
- **Lifetime is a minimum, not a pin.** Expiry is measured from the start of the request that wrote
  or last effectively used the entry; §8.2 effective use refreshes expiry under the same ordered
  receipt; a provisional lookup does not refresh. An expired entry keeps its value score but loses
  its class.
- **Intent never protects against live demand.** Running requests and protected live continuations
  always outrank all idle retention; intent never adds capacity or authorizes an allocation; PMA
  admits every byte.
- **Victim order** within a tier: (1) unexpired class, longest first; (2) §8.2 value score; (3)
  stable older-effective-use-sequence tie-break. Move only the deficit, rounded to native
  page/extent granularity; cascade only on actual overflow; tail-first partial demotion.
- **Shared pages** take the highest unexpired class of any referencing entry and move once.

## 3. Default class and default lifetime (proposal)

**Default class:** `Default`, ordered between `InMemory` and `TieredDay`.

**Rationale for the intermediate default:** caching is on by default. A request that does not ask
for retention still gets the value-based cache (the §8.2 value score governs survival), but no
strong lifetime promise. `InMemory` is reserved for explicitly short-lived scratch (e.g. an
interactive editing session that wants cache only while it is alive). `TieredDay` is the explicit
contract that the entry survives a full 24 h cycle even when idle.

**Default minimum lifetime:** the request's own lifetime plus the serving-process idle horizon,
coded as *"value-score governed; no strong retention promise"*. Concretely: the class carries no
positive minimum beyond the writing request; expiry is therefore determined entirely by the §8.2
value score and live demand. A later §8.4 amendment can raise this to a fixed minimum if
measurements show value-score-only eviction is too aggressive.

## 4. Operator ceiling on accepted classes (proposal)

- **Configuration:** optional `retention_classes_ceiling` typed intent (default: no ceiling).
- **Semantics:** when set, any request value whose class rank is **greater** than the ceiling is
  refused with a precise `invalid_request_error` (e.g. an operator who cannot honor 24 h promises
  sets ceiling `Default`; a `prompt_cache_retention: "24h"` request then fails, it is never demoted
  to `Default` silently).
- **Rationale:** a ceiling is an operator capacity/lifecycle promise, not a class remapping. Silent
  demotion would violate "recognized field with unsupported value fails".

## 5. Unknown and cache-field handling

- **Unknown API field values** (recognized field, unsupported value): `invalid_request_error`, no
  mutation, no class selection.
- **Unserved API fields** (Responses/Anthropic while not served): not interpreted; ignored (the API
  itself is not served, so there is no request to fail).
- **Absent fields** (including `prompt_cache_key` absent): default class `Default`; a keyless
  request still participates in value-based caching but cannot be re-joined by another request
  (no namespace). Keyless entries are only reachable through their own request.
- **Cache-field consistency:** a `prompt_cache_retention` without a `prompt_cache_key` is accepted
  (the class applies to the request's own keyless namespace); a `prompt_cache_key` without a
  retention value is accepted (default class). Duplicate/conflicting JSON keys are rejected by the
  parser as malformed input, never merged.

## 6. Baseline matrix freeze

The matrix below is the **frozen P0 baseline** for the retention-class/tier work. It is used by the
P3 prefix-cache/tier design and by the P8 qualification panels; changing it requires an explicit
owner decision recorded in a dated project note.

| Axis | Frozen baseline |
|---|---|
| API surface | Chat Completions only (Responses/Anthropic deferred until served) |
| Trust domains | ≥2 authenticated domains; per-domain namespace isolation; cross-domain same-key disallowed |
| Classes | `InMemory`, `Default`, `TieredDay` (rank order as §1) |
| Minimum lifetimes | `InMemory` = writing request; `Default` = value-score governed; `TieredDay` = 24 h |
| Default | `Default` (absent field) |
| Operator ceiling | optional; when set, class rank > ceiling ⇒ `invalid_request_error` |
| Key forms | with/without key; keyless reachable only via own request; duplicate keys rejected |
| Tier routes | GPU → RAM → disk (admitted route per physical tier); UMA omits redundant RAM tier |
| Move unit | native page/extent granularity; deficit-only; cascade on overflow only |
| Demotion | tail-first partial; resident head attaches, tail imports before dependent read |
| Shared pages | highest unexpired class wins; move once |
| Pressure | live demand > all idle retention; running requests > protected live continuations > idle |
| Refresh | §8.2 effective use only; provisional lookup does not refresh |
| Unsupported values | fail with precise `invalid_request_error`; never remap |

## 7. Evidence status

- **Proposal only.** No code implements these classes yet; the pooled-KV P2 work does not depend on
  them (P3 integration point).
- The §8.4 design text (§§4.3, 8.1, 8.2, 8.4) is the binding source; this note only fixes the
  unsettled value-to-class map, minimum lifetimes, default, operator ceiling and baseline matrix
  that the design explicitly deferred to P0.
