"""Frozen-history LM-head panels collected through the canonical QFN reference.

Exactly one original split-GGUF model is admitted. Each independent history uses
run_prefill_and_decode's existing cached path with explicit LM_HEAD selection;
there is no alternate forward, recurrence, tokenizer or sampling implementation.
Runtime rows/reports are local unpublished corpus candidates, never approvals.
Only small original-source metadata is digested; no model/logit payload is hashed.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Reuse existing CI metadata identity and v2 row schema authority, not a duplicate
# digest or teacher-position formula. These imports do not start native work.
sys.path.insert(0, str(ROOT/'scripts/ci'))
from model_kld_corpus import metadata_digest
from model_kld_corpus_v2 import expected_rows

ORIGINAL_PROMPT = 'The quick brown fox jumps over the lazy dog'
ORIGINAL_IDS = [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388]


SUBSET_SCHEMA = 'llaminar.qfn-accuracy-prompts.v2'


def _validate_subset(panel, *, decode_steps):
    """Check an explicit bounded subset without changing legacy panel semantics.

    Every subset case owns its decode depth and exact canonical tokenizer IDs.
    A common CLI depth is ambiguous here and must be omitted. Tokenization is
    checked through the admitted model's original tokenizer before any forward;
    these declarations cannot replace the actual tokenizer or alter its IDs.
    The shared v2 corpus remains the output/teacher-coordinate authority.
    """
    from python.reference.qwen4exp import CHECKPOINT_REVISION
    if (set(panel) != {'schema', 'tokenizer_revision', 'cases'} or
            panel['tokenizer_revision'] != CHECKPOINT_REVISION or
            type(panel['cases']) is not list or not 1 <= len(panel['cases']) <= 128 or
            decode_steps is not None):
        raise ValueError('Explicit subset needs the pinned tokenizer, 1..128 histories and per-history decode steps only')
    names, texts, token_rows = set(), set(), set()
    total = 0
    for case in panel['cases']:
        if type(case) is not dict or set(case) != {'name', 'group', 'prompt', 'expected_prompt_ids', 'decode_steps'}:
            raise ValueError('Explicit subset needs exact prompt IDs and per-history decode steps')
        for field in ('name', 'group'):
            value = case[field]
            if (type(value) is not str or not 1 <= len(value) <= 128 or value in ('.', '..') or
                    any(not (c.isascii() and (c.isalnum() or c in '-_.')) for c in value)):
                raise ValueError('Unsafe or empty subset identity')
        ids, depth, text = case['expected_prompt_ids'], case['decode_steps'], case['prompt']
        if (type(ids) is not list or not ids or
                any(type(token) is not int or not 0 <= token < 248320 for token in ids) or
                type(depth) is not int or not 1 <= depth <= 256 or
                type(text) is not str or not text or case['name'] in names or
                text in texts or tuple(ids) in token_rows):
            raise ValueError('Invalid or duplicate exact subset history')
        names.add(case['name']); texts.add(text); token_rows.add(tuple(ids))
        total += depth + 1
    if total > 4096:
        raise ValueError('Accuracy subset exceeds the existing4096 observed-row bound')
    return panel


def _validate_subset_tokenization(panel, model, vocabulary):
    """Authenticate every frozen input before any case can consume model work.

    The tokenizer is the original owner already selected by the canonical model
    factory (pinned remote revision or explicitly supplied local tokenizer). The
    shared forward helper checks the same IDs again at its actual use boundary.
    This light prepass prevents a later case's mismatch from wasting earlier
    forwards and does not admit another model or reinterpret raw token IDs.
    """
    if vocabulary != 248320 or not callable(model.tokenizer):
        raise ValueError('Explicit subset requires the actual QFN tokenizer/vocabulary')
    for case in panel['cases']:
        actual = model.tokenizer(case['prompt'], return_tensors='pt')['input_ids'][0].tolist()
        if actual != case['expected_prompt_ids']:
            raise ValueError('Subset canonical tokenization differs from frozen original IDs: '+case['name'])


def load_prompt_panel(path, *, decode_steps=None):
    """Validate the unchanged16-case v1 panel or explicit per-history v2 subset.

    Every case has a unique safe name, nonempty context group and distinct prompt.
    Legacy v1 retains its mandatory original-nine anchor; v2 subsets require no
    appended anchor. The authored file freezes text; the actual tokenizer and D+1 teacher IDs are
    frozen only in the generated candidate after original-source execution.
    An explicit CLI depth must exactly match this authored count; no request is
    silently ignored and no previously completed shorter corpus is overwritten.
    """
    panel = json.loads(Path(path).read_text(encoding='utf-8'))
    if type(panel) is dict and panel.get('schema') == SUBSET_SCHEMA:
        return _validate_subset(panel, decode_steps=decode_steps)
    if (type(panel) is not dict or panel.get('schema') != 'llaminar.qfn-accuracy-prompts.v1' or
            type(panel.get('decode_steps')) is not int or not 1 <= panel['decode_steps'] <= 256 or
            type(panel.get('cases')) is not list or len(panel['cases']) != 16):
        raise ValueError('Exactly16 authored histories with decode depth in [1,256] required')
    if len(panel['cases'])*(panel['decode_steps']+1) > 4096:
        raise ValueError('Accuracy panel exceeds the existing4096 observed-row bound')
    if decode_steps is not None and (type(decode_steps) is not int or decode_steps != panel['decode_steps']):
        raise ValueError('Explicit CLI decode steps must equal the authored panel depth')
    names, prompts = set(), set()
    for case in panel['cases']:
        if type(case) is not dict:
            raise ValueError('Malformed authored history')
        for key in ('name', 'group'):
            value = case.get(key)
            if (type(value) is not str or not value or value in ('.', '..') or
                    any(not (c.isascii() and (c.isalnum() or c in '-_.')) for c in value)):
                raise ValueError('Unsafe or empty authored case identity')
        prompt = case.get('prompt')
        if type(prompt) is not str or not prompt or case['name'] in names or prompt in prompts:
            raise ValueError('Empty/duplicate authored history')
        names.add(case['name']); prompts.add(prompt)
        if 'expected_prompt_ids' in case and (case['name'] != 'original-nine' or case['expected_prompt_ids'] != ORIGINAL_IDS):
            raise ValueError('Only the original exact nine-token anchor is pretokenized')
    first = panel['cases'][0]
    if (first['name'] != 'original-nine' or first['prompt'] != ORIGINAL_PROMPT or
            first.get('expected_prompt_ids') != ORIGINAL_IDS):
        raise ValueError('Original nine-token failure history must remain first')
    return panel


def _file_identity(path):
    """Observe an original file owner without reading or hashing its payload."""
    path = Path(path).resolve(strict=True)
    value = path.stat()
    if not path.is_file() or value.st_size <= 0:
        raise ValueError('Missing original source owner')
    return dict(path=str(path), device=value.st_dev, inode=value.st_ino,
                bytes=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def original_source(model, tokenizer_path, shared_predictor=None):
    """Bind original main owners and an optional inactive predictor by stat only.

    shared_predictor records the original file selected by the caller while MTP
    remains off. No predictor parser, mapping, tensor read, source conversion or
    forward is created here. The descriptor participates in the same source
    digest and end-of-collection identity recheck as the main split owners.
    """
    from python.reference.qwen4exp import CHECKPOINT_REVISION, TRANSFORMERS_REVISION, CONVERTER_REVISION
    parser = model.weights.parser
    paths = sorted([parser.file_path] + [child.file_path for child in parser._split_parsers])
    vocabulary = model.hf_model.config.vocab_size
    if (len(paths) != 4 or len({str(Path(p).resolve()) for p in paths}) != 4 or
            parser.metadata.get('general.architecture') != 'qwen4exp' or vocabulary != 248320):
        raise ValueError('Original four-shard QFN vocabulary/source required')
    shards = []
    for path in paths:
        stat = _file_identity(path)
        shards.append(dict(name=Path(path).name, path=stat['path'], bytes=stat['bytes'],
                           object_identity=json.dumps(stat, sort_keys=True, separators=(',', ':'))))
    source = dict(architecture='qwen4exp', main_layers=48, expert_count=512,
                vocabulary=vocabulary, reference_arithmetic='cpu_fp32',
                reference_execution='incremental_cached', shards=shards,
                tokenizer_revision=CHECKPOINT_REVISION, tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
                checkpoint_revision=CHECKPOINT_REVISION, transformers_revision=TRANSFORMERS_REVISION,
                converter_revision=CONVERTER_REVISION, gdn_output_gate='sigmoid',
                residual_width=10240, terminal_head_width=2560)
    if shared_predictor is not None:
        stat = _file_identity(shared_predictor)
        source['shared_predictor'] = dict(
            name=Path(stat['path']).name, path=stat['path'], bytes=stat['bytes'],
            object_identity=json.dumps(stat, sort_keys=True, separators=(',', ':')))
    return source


def _publish_json(path, document):
    """Atomically publish owned metadata only after its complete payload frontier."""
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(document, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def generate_accuracy_panel(args, *, tokenizer, model_factory):
    """Collect a candidate panel with one admission and guaranteed model retirement.

    Args:
        args: Existing generator arguments; subset mode omits common decode_steps.
        tokenizer: Existing optional canonical/local tokenizer owner, never rebuilt here.
        model_factory: Actual registry create_reference_model; fake only in Unit tests.
    Returns:
        Candidate workload document. It carries no topology, baseline calibration,
        approval, native proof or GREEN verdict. Root supplies those independently.
    Raises:
        ValueError: Invalid authored text, source/token geometry, finite full row,
            changed source owner or nonfresh output. Any failure leaves no complete
            reference report; the original model closes even after a failed case.
    """
    import numpy as np
    import torch
    from python.reference import PipelineStage
    from python.reference.generate_qwen35_pipeline_snapshots import run_prefill_and_decode
    panel = load_prompt_panel(args.accuracy_panel, decode_steps=args.decode_steps)
    subset = panel.get('schema') == SUBSET_SCHEMA
    output = Path(args.output)
    if output.exists():
        raise ValueError('Accuracy panel requires a fresh owned output directory')
    output.mkdir(parents=True)
    model = None
    cases, histories, token_histories = [], [], set()
    try:
        model = model_factory('qwen4exp', args.model, cache_bytes=args.cache_bytes, tokenizer=tokenizer)
        source = original_source(model, args.tokenizer, getattr(args, 'shared_predictor', None))
        vocabulary = source['vocabulary']
        if subset:
            _validate_subset_tokenization(panel, model, vocabulary)
        runtime = dict(torch_version=str(torch.__version__), intra_threads=torch.get_num_threads(),
                       interop_threads=torch.get_num_interop_threads(), torch_config=torch.__config__.show(),
                       allowed_cpu_ids=sorted(os.sched_getaffinity(0)))
        for authored in panel['cases']:
            decode_steps = authored['decode_steps'] if subset else panel['decode_steps']
            directory = output/'cases'/authored['name']
            _, prompt_ids, continuation = run_prefill_and_decode(
                model, authored['prompt'], decode_steps, directory,
                capture_stages=[PipelineStage.LM_HEAD], snapshot_final_row_only=True,
                expected_token_ids=authored.get('expected_prompt_ids'))
            if (not prompt_ids or any(type(i) is not int or not 0 <= i < vocabulary for i in prompt_ids) or
                    len(continuation) != decode_steps+1 or any(type(i) is not int or not 0 <= i < vocabulary for i in continuation) or
                    tuple(prompt_ids) in token_histories):
                raise ValueError('Missing/duplicate original prompt or D+1 teacher IDs')
            token_histories.add(tuple(prompt_ids))
            history = dict(id=authored['name'], context_group=authored['group'], prompt=authored['prompt'],
                           prompt_ids=prompt_ids, continuation_ids=continuation, decode_rows=decode_steps,
                           reference_directory=str(directory.relative_to(output)), reference_scope='final_logits_only')
            histories.append(history)
            rows = [dict(row, vocabulary=vocabulary) for row in expected_rows(dict(histories=[history]))]
            payload = directory/'logits.f32'
            with payload.open('xb') as handle:
                for index, filename in enumerate(['LM_HEAD.npy'] + [f'decode_step{i}_LM_HEAD.npy' for i in range(decode_steps)]):
                    value = np.load(directory/filename, allow_pickle=False)
                    if (value.shape != (1, 1, vocabulary) or value.dtype != np.float32 or
                            not value.flags.c_contiguous or not np.isfinite(value).all() or
                            int(np.argmax(value[0, 0])) != continuation[index]):
                        raise ValueError('Incomplete/mispaired original FP32 final vocabulary row')
                    handle.write(value.astype('<f4', copy=False).tobytes(order='C'))
            if payload.stat().st_size != (decode_steps+1)*vocabulary*4:
                raise ValueError('Incomplete reference case extent')
            cases.append(dict(history_id=history['id'], logits_file=str(payload.relative_to(output)), rows=rows))
        if original_source(model, args.tokenizer, getattr(args, 'shared_predictor', None)) != source:
            raise ValueError('Original source owner changed during reference collection')
    finally:
        if model is not None:
            model.close()
    # A close failure must not leave a complete report. Publication is therefore
    # outside finally and after the sole original model has actually retired.
    report = dict(schema='llaminar.model-kld-reference.v2', complete=True,
                  source_digest=metadata_digest(source),
                  workload_digest=metadata_digest(dict(histories=histories, vocabulary=vocabulary)),
                  cases=cases, reference_runtime=runtime)
    candidate = dict(schema='llaminar.qfn-accuracy-reference-candidate.v1', vocabulary=vocabulary,
                     source=source, histories=histories,
                     reference=dict(report='reference.json', report_digest=metadata_digest(report),
                                    format='float32_le_row_major'),
                     authored_panel_digest=metadata_digest(panel),
                     capture='LM_HEAD_final_row_only', approval='not_approved')
    _publish_json(output/'reference.json', report)
    _publish_json(output/'reference-candidate.json', candidate)
    return candidate
