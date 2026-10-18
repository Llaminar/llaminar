#!/usr/bin/env python3
"""Independent prefill/decode KL gates for the frozen dense and MoE histories.

The native diagnostic owner must first authenticate the original GGUF, frozen
prompt/forced continuation, full-vocabulary logits, captured backend path and
reset replay. This device-free policy consumes all 129 canonical KL(P||Q) rows;
its first observation follows prefill, and the next 128 follow serial decode.
Passing this policy alone is not proof of a valid native observation or speed.
"""
from dataclasses import dataclass
import math
import statistics

@dataclass(frozen=True)
class Canary:
    """Immutable model geometry and separate inclusive phase ceilings in nats."""
    model_file: str
    prompt_tokens: int
    prefill_ceiling: float
    decode_ceiling: float
    half_alpha_beta: bool
    vocabulary: int = 248320
    decode_rows: int = 128

DEFAULT_GDN_POLICY = "fp16_moe_alpha_beta"

# Same-original-GGUF FP32 references, temperature 1, no tail trimming. The MoE
# ceilings admit the selected half-alpha/beta policy. Dense limits retain the
# native dense accuracy scale; no cross-model/backend averaging. The half
# control policy improves MoE but regresses dense, so only MoE uses it by default.
CANARIES = {
    ('dense', 'cuda'): Canary('Qwen3.8-27B-IQ4_XS.gguf', 16896, .001, .09, False),
    ('dense', 'rocm'): Canary('Qwen3.8-27B-IQ4_XS.gguf', 16896, .001, .13, False),
    ('moe', 'cuda'): Canary('Qwen3.6-35B-A3B-UD-IQ3_S.gguf', 16384, .01, .41, True),
    ('moe', 'rocm'): Canary('Qwen3.6-35B-A3B-UD-IQ3_S.gguf', 16384, .05, .47, True),
}


def definition(model, backend):
    """Require an explicit supported model/backend; never borrow another limit."""
    if type(model) is not str or type(backend) is not str or (model, backend) not in CANARIES:
        raise ValueError('Explicit dense/moe and cuda/rocm canary required')
    return CANARIES[model, backend]


def evaluate_rows(rows, *, model, backend):
    """Evaluate every aligned KL row and require BOTH independent phase gates.

    Args:
        rows: Ordered canonical metric dictionaries with integer consumed_tokens
            and finite nonnegative kl_nats. No filtering or truncation is allowed.
        model: Frozen dense or moe workload identity, authenticated by the caller.
        backend: Native cuda or rocm backend identity, authenticated by the caller.
    Returns:
        Separate phase results and a conjunctive GREEN/RED numerical verdict.
    Raises:
        ValueError: Missing/extra/reordered rows, unsupported identity or invalid KL.
    """
    policy = definition(model, backend)
    if type(rows) is not list or len(rows) != 1 + policy.decode_rows:
        raise ValueError('Canary requires one prefill row and all 128 decode rows')
    values = []
    for index, row in enumerate(rows):
        if type(row) is not dict or type(row.get('consumed_tokens')) is not int or row['consumed_tokens'] != policy.prompt_tokens + index:
            raise ValueError('Canary consumed-token alignment changed')
        value = row.get('kl_nats')
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('Canary KL must be finite and nonnegative')
        values.append(value)
    phases = {}
    for phase, selected, ceiling in [('prefill', values[:1], policy.prefill_ceiling),
                                      ('decode', values[1:], policy.decode_ceiling)]:
        mean = math.fsum(selected) / len(selected)
        if not math.isfinite(mean):
            raise ValueError('Canary phase mean is nonfinite')
        phases[phase] = dict(passed=mean <= ceiling, mean_kl_nats=mean,
            ceiling_inclusive_nats=ceiling, row_count=len(selected),
            max_kl_nats=max(selected), median_kl_nats=statistics.median(selected))
    passed = all(phase['passed'] for phase in phases.values())
    return dict(passed=passed, verdict='GREEN' if passed else 'RED', phases=phases,
        model=model, backend=backend, model_file=policy.model_file,
        mean_kl_nats=math.fsum(values)/len(values), row_count=len(values),
        metric='KL(original-GGUF FP32 reference || candidate)', temperature=1.,
        units='nats', vocabulary=policy.vocabulary, row_trimming=False,
        backend_averaging=False, phase_averaging_for_acceptance=False)


def validate_report(report, *, source_revision, image, cpu_isa, corpus_manifest_digest, require_passed=True):
    """Replay complete published-image evidence, including independently red phases.

    Expected identities come from the admitted image pair. Allowing a valid red
    result lets the caller finish both ISAs; acceptance always requires green.
    All canonical rows are retained, so a fabricated summary cannot pass.
    """
    if (type(report.get('schema')) is not int or report['schema']!=1
            or report.get('complete') is not True or type(report.get('passed')) is not bool
            or report.get('source_revision')!=source_revision or report.get('image')!=image
            or report.get('corpus_manifest_digest')!=corpus_manifest_digest
            or report.get('diagnostic_only') is not False or cpu_isa not in ['AVX2','AVX512']
            or report.get('cpu_isa')!=cpu_isa):
        raise ValueError('Incomplete or mismatched published-image KL report')
    cells=report.get('cells')
    if type(cells) is not list or len(cells)!=len(CANARIES):raise ValueError('Incomplete KL cell inventory')
    seen=set();verdicts=[]
    for cell in cells:
        key=(cell.get('model'),cell.get('backend'))
        if key not in CANARIES or key in seen:raise ValueError('Duplicate/unknown KL cell')
        seen.add(key)
        if any(cell.get(k) is not True for k in ['complete','native_retired','reset_byte_equal','driver_window_clean','raw_logits_removed']) or type(cell.get('returncode')) is not int or cell['returncode']!=0:
            raise ValueError('Incomplete native KL lifetime or cleanup')
        metrics=cell.get('metrics',{});expected=evaluate_rows(metrics.get('rows'),model=key[0],backend=key[1])
        if (cell.get('passed') is not expected['passed'] or cell.get('phases')!=expected['phases']
                or any(metrics.get(k)!=v for k,v in expected.items())
                or metrics.get('corpus_manifest_digest')!=corpus_manifest_digest
                or metrics.get('full_vocabulary') is not True):
            raise ValueError('KL phase verdict differs from complete canonical rows')
        proof=cell.get('production_path',{})
        if any(proof.get(k) is not True for k in ['complete_graph_replay','prefill','decode','native_router','native_wide_projections']) or type(proof.get('half_alpha_beta_weights')) is not int or proof['half_alpha_beta_weights']!=(60 if definition(*key).half_alpha_beta else 0):
            raise ValueError('Missing captured default precision proof')
        verdicts.append(expected['passed'])
    if report['passed'] is not all(verdicts) or (require_passed and not report['passed']):
        raise ValueError('Published-image KL phase failures remain')
    return report
