"""Typed timing grades for finite native exact-policy tournaments.

Screening retains original correctness and raw clocks, but never acquires the
promotion sampling floor by being in the same corpus as a finalist. Historical
fully sampled observations keep their original meaning. This is evidence
admission, not a fitted policy or a second runtime/provenance authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

from .adapters.evidence import native_double_digest


class TimingStage(str, Enum):
    """Distinguish bounded comparison clocks from installable full timing."""
    SCREENING='screening'
    FINAL='final'


@dataclass(frozen=True)
class NativeTournamentTiming:
    """One row's validated decision in its complete original native contest."""
    stage: TimingStage
    native_candidate: str
    baseline_candidate: str
    winner_candidate: str
    finalist_reason: tuple[str,...]
    candidate_ids: tuple[str,...]
    finalist_candidate_ids: tuple[str,...]
    screen_samples_us: tuple[float,...]
    screen_digest: str
    contest_digest: str

    @property
    def promotion_grade(self) -> bool:
        """Only full finalist timing may enter the existing exact oracle."""
        return self.stage==TimingStage.FINAL

    def canonical_mapping(self, *, resolve=lambda value:value) -> dict:
        """Retain native clocks and normalize names through the existing registry."""
        return dict(schema='native_quantised_tournament_v1',timing_stage=self.stage.value,
            promotion_grade=self.promotion_grade,finalist_reason=list(self.finalist_reason),
            candidate_ids=[resolve(value) for value in self.candidate_ids],
            baseline_candidate_id=resolve(self.baseline_candidate),winner_candidate_id=resolve(self.winner_candidate),
            all_candidate_correctness_passed=True,
            finalist_candidate_ids=[resolve(value) for value in self.finalist_candidate_ids],
            screen_samples_us=list(self.screen_samples_us),screen_sorted_native_digest=self.screen_digest,
            contest_digest=self.contest_digest)


def _positive_samples(values, *, count: int) -> tuple[float,...]:
    """Reject missing, fabricated-count or non-finite native sample sequences."""
    if (not isinstance(values,list) or len(values)!=count or
            any(type(value) not in (int,float) or not math.isfinite(value) or value<=0 for value in values)):
        raise ValueError('adaptive tournament requires its original positive native screen samples')
    return tuple(float(value) for value in values)


def validate_fused_tournament(payload: Mapping, *, minimum_samples: int) -> NativeTournamentTiming:
    """Authenticate the complete source/actual-M/ISA/byte contest in fused v2.

    Every valid roster member must have passed correctness. The baseline is
    the single recorded unforced Auto observation, not a supplied oracle label.
    The selected row cannot change grade/count independently of that receipt.
    """
    from .cpu_exact_overlay import CPUExecutionExactKey, canonical_digest
    # Native adaptive v2 is always a full-floor tournament, even when a
    # diagnostic reader has a lower context sampling threshold.
    minimum_samples=max(30,minimum_samples)
    if payload.get('schema')!='cpu_execution_exact_native_v2':
        raise ValueError('adaptive expert timing requires native v2 evidence')
    key=CPUExecutionExactKey.from_scalars(payload['exact_key'])
    receipt=payload.get('tournament')
    if not isinstance(receipt,dict) or receipt.get('schema')!='native_quantised_tournament_v1':
        raise ValueError('adaptive expert timing lacks its original complete contest')
    expected=dict(backend='cpu',source_format=payload['source_format'],purpose=payload['purpose'],
        m=key.rows,n=key.physical_n,k=key.k,threads=key.workers,build_isa=key.build_isa,runtime_isa=key.runtime_isa,
        holdout=payload['holdout'],expert_rows=payload['expert_rows'],target_descriptor=payload['target_descriptor'],
        exact_key=list(key.scalars()))
    if any(receipt.get(name)!=value for name,value in expected.items()):
        raise ValueError('adaptive contest changed its original source/actual-M/arithmetic/ISA identity')
    if receipt.get('shape')!=payload.get('shape'):
        raise ValueError('adaptive contest changed its original named geometry')
    roster=receipt.get('candidate_ids');finalists=receipt.get('finalist_candidate_ids')
    if (not isinstance(roster,list) or not roster or len(set(roster))!=len(roster) or
            not isinstance(finalists,list) or not finalists or len(set(finalists))!=len(finalists) or
            set(finalists)-set(roster) or receipt.get('baseline_candidate_id') not in finalists or
            receipt.get('winner_candidate_id') not in finalists or receipt.get('baseline_auto_calls')!=1 or
            not receipt.get('baseline_auto_route_records')):
        raise ValueError('adaptive contest lacks its complete forceable roster or actual Auto baseline')
    routes=receipt['baseline_auto_route_records'];total=0;targets=0
    target_tags=dict(m=str(key.rows),n=str(key.physical_n),policy_n=str(key.policy_n),k=str(key.k),
        codebook=str(key.source_codebook),purpose=payload['purpose'],isa=('AVX2','AVX512')[key.runtime_isa])
    if not isinstance(routes,list):raise ValueError('adaptive baseline route observations are malformed')
    for route in routes:
        if not isinstance(route,dict) or type(route.get('count')) is not int or route['count']<=0 or not isinstance(route.get('tags'),dict):
            raise ValueError('adaptive baseline lacks original positive native route records')
        total+=route['count'];tags=route['tags']
        if all(tags.get(name)==value for name,value in target_tags.items()):
            targets+=route['count']
            if tags.get('requested_verifier_policy')!='Auto' or tags.get('effective_verifier_policy')!=receipt['baseline_candidate_id']:
                raise ValueError('adaptive baseline is not the actual unforced native policy')
    if total!=len(payload['descriptor_keys']) or targets<=0:
        raise ValueError('adaptive baseline route records do not cover original live descriptors')
    expected_roster={'Pairwise','WideRows'} if key.runtime_isa==1 and key.rows>=3 else {'Pairwise'}
    if set(roster)!=expected_roster:
        raise ValueError('adaptive expert roster omitted or invented an original forceable candidate')
    decisions=receipt.get('candidates')
    if not isinstance(decisions,list) or len(decisions)!=len(roster):
        raise ValueError('adaptive contest omitted a screened candidate decision')
    indexed={}
    for row in decisions:
        candidate=row.get('candidate_id')
        if candidate in indexed or candidate not in roster or row.get('correctness_pass') is not True:
            raise ValueError('adaptive tournament cannot eliminate a numerical failure or missing candidate')
        stage=TimingStage(row.get('timing_stage'))
        grade=stage==TimingStage.FINAL
        if type(row.get('promotion_grade')) is not bool or row['promotion_grade']!=grade or (candidate in finalists)!=grade:
            raise ValueError('adaptive tournament grade disagrees with actual finalist decision')
        samples=_positive_samples(row.get('screen_samples_us'),count=3)
        if native_double_digest(sorted(samples))!=row.get('screen_sorted_native_digest'):
            raise ValueError('adaptive screening sample digest changed')
        reasons=tuple(filter(None,str(row.get('finalist_reason','')).split('|')))
        allowed={'screen_winner','installed_baseline','credible_runner_up','row_regime_anchor','screened_out'}
        if (len(set(reasons))!=len(reasons) or set(reasons)-allowed or not reasons or
                (grade and 'screened_out' in reasons) or (not grade and reasons!=('screened_out',))):
            raise ValueError('adaptive finalist reason is incomplete or incompatible')
        if ('installed_baseline' in reasons)!=(candidate==receipt['baseline_candidate_id']):
            raise ValueError('adaptive baseline marker disagrees with actual Auto receipt')
        count=row.get('final_sample_count')
        if type(count) is not int or (grade and count<minimum_samples) or (not grade and count!=0):
            raise ValueError('adaptive finalist lacks original full timing floor')
        indexed[candidate]=(stage,reasons,samples,row)
    candidate=payload['candidate']
    if candidate not in indexed:raise ValueError('adaptive row is absent from its original contest')
    stage,reasons,screen,row=indexed[candidate]
    if (payload.get('timing_stage')!=stage.value or payload.get('promotion_grade')!=(stage==TimingStage.FINAL) or
            type(payload.get('promotion_grade')) is not bool or payload.get('finalist_reason')!=row['finalist_reason'] or
            payload.get('screen_samples_us')!=list(screen) or payload.get('screen_repeat_count')!=1):
        raise ValueError('adaptive result row changed its retained timing grade/screen clocks')
    timing=payload['timing'];final=stage==TimingStage.FINAL
    if (payload.get('configured_full_sample_count',0)<minimum_samples or
            type(payload.get('configured_full_repeat_count')) is not int or payload['configured_full_repeat_count']<=0 or
            timing['sample_count']!=(row['final_sample_count'] if final else 3) or
            timing['repeat_count']!=(payload['configured_full_repeat_count'] if final else 1) or
            (final and timing['warmup_count']<5) or
            (not final and (timing['warmup_count']!=0 or payload['raw_samples_us']!=list(screen)))):
        raise ValueError('adaptive row relabeled screening clocks as fully sampled evidence')
    return NativeTournamentTiming(stage,candidate,receipt['baseline_candidate_id'],receipt['winner_candidate_id'],reasons,tuple(roster),
        tuple(finalists),screen,row['screen_sorted_native_digest'],canonical_digest(receipt))



def validate_adaptive_tournament_witness(evidence: Mapping) -> None:
    """Validate normalized retained evidence before an exact-owner decision."""
    if (not isinstance(evidence,dict) or evidence.get('schema')!='native_quantised_tournament_v1' or
            evidence.get('timing_stage') not in {stage.value for stage in TimingStage} or
            type(evidence.get('promotion_grade')) is not bool or
            evidence['promotion_grade']!=(evidence['timing_stage']=='final') or
            evidence.get('all_candidate_correctness_passed') is not True):
        raise ValueError('incomplete or incompatible adaptive timing grade')
    roster=evidence.get('candidate_ids');finalists=evidence.get('finalist_candidate_ids')
    if (not isinstance(roster,list) or not roster or any(not isinstance(value,str) or not value for value in roster) or
            len(set(roster))!=len(roster) or not isinstance(finalists,list) or not finalists or
            len(set(finalists))!=len(finalists) or set(finalists)-set(roster) or
            evidence.get('baseline_candidate_id') not in finalists or evidence.get('winner_candidate_id') not in finalists):
        raise ValueError('adaptive witness lost its complete native roster/baseline/finalists')
    samples=_positive_samples(evidence.get('screen_samples_us'),count=3)
    if native_double_digest(sorted(samples))!=evidence.get('screen_sorted_native_digest'):
        raise ValueError('adaptive witness screening clocks changed')
    digest=evidence.get('contest_digest')
    if not isinstance(digest,str) or not digest.startswith('sha256:') or len(digest)!=71:
        raise ValueError('adaptive witness lacks original contest identity')
    reasons=evidence.get('finalist_reason')
    allowed={'screen_winner','installed_baseline','credible_runner_up','row_regime_anchor','screened_out'}
    if (not isinstance(reasons,list) or not reasons or len(set(reasons))!=len(reasons) or set(reasons)-allowed or
            (evidence['promotion_grade'] and 'screened_out' in reasons) or
            (not evidence['promotion_grade'] and reasons!=['screened_out'])):
        raise ValueError('adaptive witness timing grade/reason disagree')


def is_promotion_grade(observation) -> bool:
    """Keep old rows eligible; new screening clocks are never winner evidence."""
    evidence=observation.physical_launch_evidence.get('tournament')
    if evidence is None:return True
    validate_adaptive_tournament_witness(evidence)
    if observation.supported and observation.forced_route_ok and (
            not observation.numerical_correctness or observation.mismatch_count or not observation.repeat_equal):
        raise ValueError('adaptive tournament cannot retain numerical failure as a screened loser')
    return evidence['promotion_grade']
