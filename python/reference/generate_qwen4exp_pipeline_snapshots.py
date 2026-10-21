#!/usr/bin/env python3
"""Generate exact-GGUF QFN main and explicit paired Dense0 reference snapshots.

The existing canonical main writer remains the sole main-model oracle. Optional
sidecar expansion reads its authenticated wide trajectory and the explicitly
selected original predictor; additive branches never rerun or overwrite main.
No numerical MTP matrix/support declaration is made by this entrypoint.
An explicit fixed-history accuracy panel reuses one admitted main model and the
same cached helper, publishing only final FP32 LM-head rows, never full-stage
coverage or a corpus approval. Explicit --frozen-history selects one existing
Pending teacher trajectory for full-stage capture through the same cached helper.
The ordinary single-prompt path is unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference import create_reference_model
from python.reference.generate_qwen35_pipeline_snapshots import run_prefill_and_decode, write_metadata
from python.reference.qwen4exp import (
    CHECKPOINT_REVISION, TRANSFORMERS_REVISION, CONVERTER_REVISION, NativeSliceWeights,
)
from python.reference.qwen4exp_mtp_snapshots import (
    bind_dense0_sources, load_main_trajectory, write_dense0_snapshots, publish_dense0_metadata,
    authenticate_dense0_metadata,
)
from python.reference.mtp_sidecar_reference import normalize_mtp_branch_override_batches


def parse_args(argv=None):
    """Validate explicit source/depth/branch intent before model or artifact work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='exact main GGUF or first split')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', default='The quick brown fox jumps over the lazy dog')
    parser.add_argument('--decode-steps', type=int, default=None, help='ordinary/v1 default3; omit for explicit per-history v2 accuracy subsets')
    parser.add_argument('--cache-bytes', type=int, default=1 << 30)
    parser.add_argument('--accuracy-panel', type=Path, help='LM_HEAD-only authored panel: v1 fixed16/common depth or v2 exact subset/per-history depths; no MTP')
    parser.add_argument('--frozen-history', type=Path, help='existing v2 corpus manifest; capture exactly one frozen history with all stages')
    parser.add_argument('--history-id', help='exact original history ID selected from --frozen-history')
    parser.add_argument('--tokenizer', help='local tokenizer directory (otherwise pinned HF revision)')
    parser.add_argument('--shared-predictor', type=Path, help='explicit original Dense0/PerBranch MTP GGUF; panel/frozen-history records inactive identity only')
    parser.add_argument('--mtp-sidecar-snapshots', action='store_true')
    parser.add_argument('--mtp-max-draft-depth', type=int, default=15)
    parser.add_argument('--mtp-sidecar-only', action='store_true', help='reuse authenticated main trajectory, without main forward')
    parser.add_argument('--mtp-branch-overrides', type=Path, help='shared typed additive branch JSON request')
    args = parser.parse_args(argv)
    if args.decode_steps is None and args.accuracy_panel is None:
        args.decode_steps = 3  # Preserve every ordinary/MTP/frozen-history caller's original default.
    if (args.decode_steps is not None and args.decode_steps < 0) or args.cache_bytes < 0:
        parser.error('decode steps and decoded-weight cache bytes must be nonnegative')
    if not 1 <= args.mtp_max_draft_depth <= 15:
        parser.error('MTP maximum depth must be in [1,15]')
    if ((args.mtp_sidecar_snapshots and args.shared_predictor is None) or
            (args.shared_predictor is not None and not args.mtp_sidecar_snapshots and args.accuracy_panel is None and args.frozen_history is None)):
        parser.error('MTP snapshots require exactly one explicit --shared-predictor')
    if (args.mtp_sidecar_only or args.mtp_branch_overrides is not None) and not args.mtp_sidecar_snapshots:
        parser.error('sidecar-only/branch input requires explicit paired-source MTP snapshots')
    if args.mtp_branch_overrides is not None and not args.mtp_sidecar_only:
        parser.error('additive MTP branches require --mtp-sidecar-only')
    if args.mtp_sidecar_snapshots and args.decode_steps == 0:
        parser.error('MTP snapshots require a nonempty canonical decode trajectory')
    if args.accuracy_panel is not None and (args.mtp_sidecar_snapshots or
                                            args.prompt != 'The quick brown fox jumps over the lazy dog'):
        parser.error('accuracy panel requires no separate prompt and no MTP sidecar')
    if args.accuracy_panel is not None:
        from python.reference.qwen4exp_accuracy_panel import load_prompt_panel, SUBSET_SCHEMA
        try:
            panel = load_prompt_panel(args.accuracy_panel, decode_steps=args.decode_steps)
            if args.decode_steps is None and panel['schema'] != SUBSET_SCHEMA:
                args.decode_steps = 3  # Legacy v1's implicit CLI default remains exactly3.
                load_prompt_panel(args.accuracy_panel, decode_steps=args.decode_steps)
        except ValueError as error:
            parser.error(str(error))
    if bool(args.frozen_history) != bool(args.history_id):
        parser.error('--frozen-history and --history-id require each other')
    if args.frozen_history is not None:
        if args.accuracy_panel is not None or args.mtp_sidecar_snapshots or args.mtp_sidecar_only or args.mtp_branch_overrides is not None:
            parser.error('frozen FullStages history is separate from LM-only panel and MTP sidecar')
        try:
            if args.output.exists():
                raise ValueError('Frozen FullStages history requires a fresh output directory')
            load_frozen_history(args)
        except (ValueError, OSError, KeyError, TypeError) as error:
            parser.error(str(error))
    return args


def load_frozen_history(args):
    """Check one original manifest/history/source without model admission or payload reads.

    Returns the unchanged source manifest and one selected history. The shared
    corpus validator owns schema/teacher geometry; original stat comparison
    prevents same-size source substitution. No topology is selected or modified.
    """
    from python.reference.qwen4exp_accuracy_panel import _file_identity
    from model_kld_corpus_v2 import validate_manifest
    path = args.frozen_history
    size = path.stat().st_size
    if not 0 < size <= 16*1024*1024:
        raise ValueError('Frozen history manifest exceeds bounded metadata extent')
    with path.open('rb') as stream:
        raw = stream.read(16*1024*1024+1)
    if len(raw) != size:
        raise ValueError('Frozen history manifest changed while reading bounded metadata')
    manifest = json.loads(raw.decode('utf-8'))
    validate_manifest(manifest)
    histories = [history for history in manifest['histories'] if history['id'] == args.history_id]
    if len(histories) != 1:
        raise ValueError('Missing original frozen history ID')
    history = histories[0]
    source = manifest['source']
    if (history['decode_rows'] != args.decode_steps or
            Path(args.model).resolve(strict=True) != Path(source['shards'][0]['path']).resolve(strict=True) or
            args.tokenizer != source.get('tokenizer_path') or
            args.prompt not in ('The quick brown fox jumps over the lazy dog',history['prompt'])):
        raise ValueError('Frozen source/tokenizer/prompt/decode work differs from explicit request')
    paired = source.get('shared_predictor')
    if args.shared_predictor is not None and (not paired or str(args.shared_predictor) != paired['path']):
        raise ValueError('Frozen inactive predictor source differs from explicit request')
    for descriptor in source['shards'] + ([paired] if paired else []):
        actual = _file_identity(descriptor['path'])
        original = json.loads(descriptor['object_identity'])
        if (type(original) is not dict or set(original) != set(actual) or
                any(type(original[key]) is not type(value) or original[key] != value for key,value in actual.items()) or
                descriptor['bytes'] != actual['bytes'] or descriptor['name'] != Path(actual['path']).name):
            raise ValueError('Frozen original source stat identity changed')
    return manifest, history


def generate_frozen_history(args, *, tokenizer, model_factory):
    """Capture one admitted model's complete cached stage trajectory and retire it.

    Both source and immutable manifest are rechecked after the last observation.
    Complete metadata is published only after successful original model close.
    The optional inactive predictor is stat metadata only; no predictor is parsed.
    """
    from python.reference.qwen4exp_accuracy_panel import original_source, metadata_digest
    if args.output.exists():
        raise ValueError('Frozen FullStages history requires a fresh output directory')
    manifest, history = load_frozen_history(args)
    source = manifest['source']
    paired = source.get('shared_predictor',{}).get('path')
    model = None
    try:
        model = model_factory('qwen4exp',args.model,cache_bytes=args.cache_bytes,tokenizer=tokenizer)
        if original_source(model,args.tokenizer,paired) != source:
            raise ValueError('Actual admitted reference source differs from frozen original')
        count, token_ids, teachers = run_prefill_and_decode(model,history['prompt'],args.decode_steps,args.output,
            expected_token_ids=history['prompt_ids'],forced_decode_tokens=history['continuation_ids'])
        after, _ = load_frozen_history(args)
        if after != manifest or original_source(model,args.tokenizer,paired) != source:
            raise ValueError('Frozen history/source changed during full-stage collection')
    finally:
        if model is not None:
            model.close()
    write_metadata(args.output,args.model,model,history['prompt'],token_ids,args.decode_steps,teachers,
        extra_metadata_lines=reference_fields()+[
            'qwen4exp_frozen_corpus_digest: '+metadata_digest(manifest),
            'qwen4exp_frozen_source_digest: '+metadata_digest(source),
            'qwen4exp_frozen_history_id: '+history['id'],
            'qwen4exp_teacher_policy: explicit_frozen_continuation_ids'])
    return count, token_ids, teachers


def reference_fields():
    """Retain the original main-reference schema and pinned equation identities."""
    return ['qwen4exp_reference_schema: 1',
            f'qwen4exp_checkpoint_revision: {CHECKPOINT_REVISION}',
            f'qwen4exp_transformers_revision: {TRANSFORMERS_REVISION}',
            f'qwen4exp_converter_revision: {CONVERTER_REVISION}',
            'qwen4exp_integer_checkpoint_dtype: int64', 'qwen4exp_gdn_output_gate: sigmoid',
            'qwen4exp_residual_width: 10240', 'qwen4exp_terminal_head_width: 2560']


def main(argv=None):
    """Use canonical main generation, then optional authenticated sidecar expansion."""
    args = parse_args(argv)
    branches = None
    if args.mtp_branch_overrides is not None:
        branches = normalize_mtp_branch_override_batches(json.loads(
            args.mtp_branch_overrides.read_text(encoding='utf-8')))
        # Existing canonical predictor authority precedes tokenizer/source setup,
        # main-row loads and every possible additive publication.
        authenticate_dense0_metadata(args.output, args.shared_predictor,
                                    args.mtp_max_draft_depth, args.decode_steps)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.frozen_history is not None:
        return generate_frozen_history(args,tokenizer=tokenizer,model_factory=create_reference_model)
    if args.accuracy_panel is not None:
        from python.reference.qwen4exp_accuracy_panel import generate_accuracy_panel
        return generate_accuracy_panel(args, tokenizer=tokenizer,
                                       model_factory=create_reference_model)
    model = None
    main_weights = None
    predictor = None
    try:
        if args.mtp_sidecar_only:
            if not (args.output/'metadata.txt').is_file():
                raise ValueError('sidecar-only requires existing canonical main metadata')
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.8-Flash-Next', revision=CHECKPOINT_REVISION)
            token_ids = tokenizer(args.prompt, return_tensors='pt')['input_ids'][0].tolist()
            # Authenticate trajectory before parsing predictor or accessing its payload.
            trajectory = load_main_trajectory(args.output, args.model, args.prompt, token_ids, args.decode_steps)
            main_weights = NativeSliceWeights(args.model, args.cache_bytes)
        else:
            model = create_reference_model('qwen4exp', args.model, cache_bytes=args.cache_bytes, tokenizer=tokenizer)
            main_weights = model.weights
        sidecar = None
        if args.mtp_sidecar_snapshots:
            predictor = NativeSliceWeights(args.shared_predictor, args.cache_bytes)
            sidecar = bind_dense0_sources(main_weights, predictor)
        if model is not None:
            _, token_ids, decode_tokens = run_prefill_and_decode(model, args.prompt, args.decode_steps, args.output)
            write_metadata(args.output, args.model, model, args.prompt, token_ids,
                           args.decode_steps, decode_tokens, extra_metadata_lines=reference_fields())
        if sidecar is not None:
            if model is not None:
                trajectory = load_main_trajectory(args.output, args.model, args.prompt, token_ids, args.decode_steps)
            count = write_dense0_snapshots(sidecar, trajectory, args.output, args.mtp_max_draft_depth, branches)
            if not count:
                raise ValueError('paired Dense0 request produced no authenticated predictor snapshots')
            if branches is None:
                publish_dense0_metadata(args.output, args.shared_predictor, args.mtp_max_draft_depth, args.decode_steps)
    finally:
        if predictor is not None:
            predictor.close()
        if model is not None:
            model.close()
        elif main_weights is not None:
            main_weights.close()


if __name__ == '__main__':
    main()
