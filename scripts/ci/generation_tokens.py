"""Exact terminal-token observations for production generation regression.

This module neither generates nor updates baselines. It reads the optional
token-ID response from the ordinary HTTP handler and reports the first exact
difference, including prompt tokenization and termination. A passing token
comparison is not a KL/cosine certificate: distribution checks are separate.
No dependency on HF, snapshots, model files, or accelerator libraries is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class GenerationWorkload:
    """Cell-owned evidence budget, not an inferred MTP-depth heuristic.

    The C++ matrix owns the canonical horizon. Python validates its exported
    shape and measures a single response against it, without a second default
    table or permission to accumulate several short completions.
    """
    max_tokens: int
    minimum_completion_tokens: int

    def __post_init__(self) -> None:
        """Reject unusable budgets before starting a server or accepting output."""
        if (type(self.max_tokens) is not int or type(self.minimum_completion_tokens) is not int
                or not 0 < self.minimum_completion_tokens <= self.max_tokens):
            raise ValueError("generation workload requires positive minimum <= maximum integer tokens")

    @classmethod
    def from_record(cls, record: dict) -> GenerationWorkload:
        """Require the canonical exported workload; missing metadata is not a default."""
        runtime = record.get("runtime") if isinstance(record, dict) else None
        profile = runtime.get("generation") if isinstance(runtime, dict) else None
        if not isinstance(profile, dict):
            raise ValueError("canonical cell omitted its generation workload; rebuild its matrix")
        return cls(profile.get("max_tokens"), profile.get("minimum_completion_tokens"))

    def observe(self, response: dict) -> TokenTrace:
        """Validate one complete request, including early-stop evidence length.

        Matching short outputs are insufficient evidence, even when both runs
        stop naturally at EOS. The caller must preserve this failure, not extend
        the output with another request or compare only the common prefix.
        """
        trace = TokenTrace.from_response(response, requested_max_tokens=self.max_tokens)
        if len(trace.completion) < self.minimum_completion_tokens:
            raise ValueError(f"insufficient continuous generation evidence: observed {len(trace.completion)} "
                             f"committed tokens, require {self.minimum_completion_tokens} in one request")
        return trace


def _token_ids(value: object, field: str) -> tuple[int, ...]:
    """Reject text re-encoding, booleans, floats, and invalid runtime token IDs."""
    if not isinstance(value, list) or any(type(token) is not int or token < 0 or token > 2147483647
                                          for token in value):
        raise ValueError(f"{field} must contain exact nonnegative int32 token IDs")
    return tuple(value)


@dataclass(frozen=True)
class TokenTrace:
    """One completed request's actual prompt, committed output, and termination."""
    prompt: tuple[int, ...]
    completion: tuple[int, ...]
    finish_reason: str

    @classmethod
    def from_response(cls, response: dict, *, requested_max_tokens: int) -> TokenTrace:
        """Admit an ordinary non-streaming response with consistent usage counts.

        The server returns EOS and forced thinking continuations in the token
        vector, even when text framing hides them. An early stop remains a
        trace to compare, not permission to compare only a matching prefix.
        """
        if type(requested_max_tokens) is not int or requested_max_tokens <= 0:
            raise ValueError("requested token budget must be a positive integer")
        if not isinstance(response, dict) or response.get("object") != "chat.completion" or "error" in response:
            raise ValueError("expected a successful non-streaming chat completion")
        ids = response.get("token_ids")
        if not isinstance(ids, dict):
            raise ValueError("response omitted requested token IDs; never reconstruct them from text")
        prompt = _token_ids(ids.get("prompt"), "prompt")
        completion = _token_ids(ids.get("completion"), "completion")
        if not prompt or len(completion) > requested_max_tokens:
            raise ValueError("empty prompt or completion beyond admitted response budget")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("token regression requires exactly one response choice")
        finish = choices[0].get("finish_reason")
        if finish not in ("length", "stop", "tool_calls"):
            raise ValueError("response has no supported terminal finish reason")
        if finish == "length" and len(completion) != requested_max_tokens:
            raise ValueError("length-terminated response did not fill its token budget")
        usage = response.get("usage")
        counts = {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                  "total_tokens": len(prompt) + len(completion)}
        if not isinstance(usage, dict) or any(type(usage.get(name)) is not int or usage[name] != count
                                               for name, count in counts.items()):
            raise ValueError("response token IDs and usage counts disagree")
        return cls(prompt, completion, finish)


class TokenTracePhase(str, Enum):
    """The first mismatch's causal boundary, without guessing a numerical cause."""
    PROMPT = "prompt"
    COMPLETION = "completion"
    TERMINATION = "termination"


@dataclass(frozen=True)
class TokenMismatch:
    """First differing token; None denotes an exhausted stream or no position."""
    phase: TokenTracePhase
    position: int | None
    expected: int | str | None
    observed: int | str | None


def compare_tokens(expected: TokenTrace, observed: TokenTrace) -> TokenMismatch | None:
    """Compare all positions and termination, never just their common prefix.

    Prompt drift is reported first because completions from different encoded
    inputs are not an equivalent workload. Finish reasons remain independent
    evidence even when both vectors contain identical IDs.
    """
    for phase, left, right in ((TokenTracePhase.PROMPT, expected.prompt, observed.prompt),
                               (TokenTracePhase.COMPLETION, expected.completion, observed.completion)):
        for index in range(max(len(left), len(right))):
            before = left[index] if index < len(left) else None
            after = right[index] if index < len(right) else None
            if before != after:
                return TokenMismatch(phase, index, before, after)
    if expected.finish_reason != observed.finish_reason:
        return TokenMismatch(TokenTracePhase.TERMINATION, None, expected.finish_reason, observed.finish_reason)
    return None
