"""Shared complete-vocabulary, temperature-one accuracy observations.

The original long-canary FP64 KL(reference || candidate) equation lives here so
short diagnostics and corpus scoring cannot silently choose different tails or
temperatures. This module computes observations only: it owns no thresholds,
model/reference loading, token timeline, backend policy or acceptance verdict.
Callers must authenticate original row positions and vocabulary before calling.
Ties always prefer the smaller token ID, including at the top-five boundary.
"""
import math


def _row(values, vocabulary):
    """Validate one complete real numeric row before converting its FP64 view.

    Masked, boolean, complex, textual and object arrays are not logit evidence.
    No squeezing, flattening, vocabulary trimming or nonfinite repair is allowed.
    The declared vocabulary independently detects equally truncated input rows.
    """
    import numpy as np
    if np.ma.isMaskedArray(values):
        raise ValueError('Masked logit evidence is incomplete')
    try:
        original = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid complete logit row') from error
    if original.ndim != 1 or original.shape != (vocabulary,) or original.dtype.kind not in 'fiu':
        raise ValueError('Invalid complete logit row geometry or numeric type')
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.asarray(original, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite complete logit row')
    return result


def _log_probabilities(logits):
    """Normalize in FP64 with the original canary's max-shifted equation.

    Exponent underflow to zero is representational, not an arbitrary probability
    cutoff. A finite row whose log range cannot be represented in FP64 fails
    explicitly rather than manufacturing finite KL from an undefined product.
    """
    import numpy as np
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        shifted = logits - np.max(logits)
        normalized = shifted - np.log(np.exp(shifted).sum())
    if not np.isfinite(normalized).all():
        raise ValueError('Logit range exceeds finite FP64 normalization')
    return normalized


def _top_ids(logits, count):
    """Select a small top set in linear work with explicit token-ID tie order.

    Partition finds only the cutoff value; all strictly greater entries and the
    lowest-ID ties fill the exact selected width. Sorting only those few entries
    avoids a full-vocabulary sort for every corpus row.
    """
    import numpy as np
    cutoff = np.partition(logits, logits.size - count)[logits.size - count]
    higher = np.flatnonzero(logits > cutoff)
    ties = np.flatnonzero(logits == cutoff)[:count - higher.size]
    chosen = np.concatenate((higher, ties))
    return chosen[np.lexsort((chosen, -logits[chosen]))]


def full_vocabulary_metrics(reference, candidate, *, vocabulary):
    """Observe aligned complete rows without applying an accuracy policy.

    Args:
        reference: Original reference logits, one real finite numeric vector.
        candidate: Candidate logits for the same original position/vocabulary.
        vocabulary: Independently authenticated integer extent, at least two.
    Returns:
        JSON-compatible observations. Legacy keys ``kl_nats``,
        ``total_variation`` and ``top1_same`` retain their meanings. Additional
        fields describe deterministic top-five intersection, reference entropy,
        reference top-one logit/probability margins, and probability errors.
        ``top5_width`` records available vocabulary slots; ``top5_overlap`` is
        intersection/5 only when vocabulary>=5; it is None for tiny legacy
        fixture vocabularies, which cannot supply a complete top-five set.
        Probability error at reference top1 is signed candidate minus reference;
        max/mean absolute errors compare all original vocabulary probabilities.
    Raises:
        ValueError: Incomplete/malformed/nonfinite rows, invalid vocabulary, or
            an arithmetic result not representable as finite FP64 evidence.
    """
    import numpy as np
    if type(vocabulary) is not int or vocabulary < 2:
        raise ValueError('Explicit complete vocabulary of at least two required')
    p = _row(reference, vocabulary)
    q = _row(candidate, vocabulary)
    lp, lq = _log_probabilities(p), _log_probabilities(q)
    probability = np.exp(lp)
    candidate_probability = np.exp(lq)
    # Preserve the original canary equation and reduction for ordinary finite
    # ranges. There is no positive-probability threshold or tail truncation.
    with np.errstate(over='ignore', invalid='ignore'):
        divergence = float(np.dot(probability, lp - lq))
    if not math.isfinite(divergence) or divergence < -1e-12:
        raise ValueError('Invalid KL arithmetic')
    delta = candidate_probability - probability
    reference_top = _top_ids(p, min(5, vocabulary))
    candidate_top = _top_ids(q, min(5, vocabulary))
    first, second = map(int, reference_top[:2])
    with np.errstate(over='ignore', invalid='ignore'):
        logit_margin = float(p[first] - p[second])
    if not math.isfinite(logit_margin):
        raise ValueError('Top1 margin exceeds finite FP64 arithmetic')
    return dict(
        kl_nats=max(0., divergence),
        total_variation=float(np.abs(probability - candidate_probability).sum() / 2),
        top1_same=bool(np.argmax(p) == np.argmax(q)),
        top5_overlap=(len(set(map(int, reference_top)) & set(map(int, candidate_top))) / 5.
                      if vocabulary >= 5 else None),
        top5_width=min(5, vocabulary),
        reference_top5_ids=reference_top.tolist(),
        candidate_top5_ids=candidate_top.tolist(),
        reference_entropy_nats=float(-np.dot(probability, lp)),
        reference_top1_logit_margin=logit_margin,
        reference_top1_probability_margin=float(probability[first] - probability[second]),
        reference_top1_probability=float(probability[first]),
        candidate_probability_at_reference_top1=float(candidate_probability[first]),
        reference_top1_probability_error=float(delta[first]),
        max_probability_error=float(np.max(np.abs(delta))),
        mean_absolute_probability_error=float(np.mean(np.abs(delta))))
