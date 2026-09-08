# CPU PyTorch/Hugging Face reference

## Authority and purpose

The Python project under `python/reference/` is the independent numerical
oracle for model parity. It runs on CPU in FP32, reconstructs the Hugging Face
model from the exact production GGUF, publishes NumPy checkpoint tensors, and
writes complete identity metadata last. Native tests consume those immutable
snapshots; Python is not part of the production inference path.

Use the current implementation as authority. `python/reference/README.md` can
lag code, so verify flags and schemas in the relevant generator and
`snapshot_metadata.py` before changing them.

## Project map

- `base.py`: `AbstractReferenceModel` and `HuggingFaceReferenceModel`; chooses
  GGUF or ordinary Hugging Face loading, installs/removes forward hooks, and
  provides full prefill plus incremental cached decode.
- `registry.py`: maps architecture names to registered reference model
  factories. Add model families here instead of branching every generator.
- `loaders/gguf_parser.py`: parses GGUF metadata and tensor records.
- `loaders/dequantize.py`: dequantizes supported GGUF tensor formats.
- `loaders/tensor_name_mapper.py`: maps GGUF tensor names to Hugging Face state
  dictionary conventions.
- `loaders/gguf_loader.py`: orchestrates parsing, CPU/FP32 dequantization,
  architecture-specific tensor transforms, and state-dict construction.
- `pipeline_stages.py`: canonical checkpoint/stage vocabulary.
- `qwen.py`, `qwen35.py`, `qwen35_moe.py`, and peers: architecture-specific
  Hugging Face construction, weight transforms, hooks, and snapshot rules.
- `generate_*_pipeline_snapshots.py`: command-line entrypoints used by C++
  fixtures to generate prefill and incremental-decode packs.
- `snapshot_metadata.py`: lightweight model descriptors, small-input identity
  hashing, schema validation, and atomic
  `metadata.txt` publication.
- `tests/` and loader tests: device-free regression coverage for metadata,
  parsing, dequantization, and reference behavior.

## How a reference pack is produced

1. Parse the exact `.gguf` supplied to the native parity configuration.
2. Create the matching Transformers configuration/model on CPU. Reference
   implementations select eager attention when hooks need materialized stage
   boundaries.
3. Dequantize GGUF weights to FP32 and transform layouts into the Hugging Face
   state dictionary. This is not a download of a similarly named checkpoint.
4. Tokenize the exact prompt and run evaluation/inference mode with no
   gradients.
5. Capture embedding, every supported per-layer stage, final norm, LM head,
   and architecture-specific data. Incremental decode reuses the Hugging Face
   cache so its state transition matches autoregressive inference.
6. Save each checkpoint as an individual `.npy` file. Decode names carry their
   step; MTP sidecars carry depth/sidecar identity required by the schema.
7. Flush snapshot files, then atomically publish `metadata.txt` with fsync and
   rename. Metadata is the commit record; a directory without valid metadata is
   not a complete pack.

`Qwen35MoEReferenceModel` is especially important for sparse parity. It builds
the eager Hugging Face `Qwen3_5MoeForCausalLM`, performs the GGUF-to-HF expert
weight transforms, fuses expert gate/up weights in the reference layout, and
captures the full post-softmax router distribution plus routed expert, shared
expert, shared gate, and combined outputs. Its MTP path recursively snapshots
sidecar stages rather than treating the final accepted token as sufficient.

## Authentication contract

The current metadata identity records a supported schema and binds:

- engine `pytorch`, device `cpu`, and dtype `float32`;
- the declared GGUF filename and byte length for newly generated packs;
- SHA-256 of the exact UTF-8 prompt bytes;
- the nonempty token sequence and its canonical SHA-256;
- requested decode depth and generated decode-token identity;
- required boundary/stage availability;
- architecture-specific sidecar schema, including Qwen3.6 MoE MTP schema 5
  where applicable.

Campaign registration binds the model and reference directory as one typed
case. Staging binds every split shard to stable source and immutable tmpfs
destination identities, and newly generated metadata adds a filename/size
diagnostic. The C++ fixture never rereads the GGUF just to hash it: full
checkpoint comparisons prove weight-content equivalence and make a wrong model
fail mathematically. Rank zero alone regenerates an incomplete pack, and all
ranks wait for the result. Never accept an empty token list, too few decode
steps, or metadata written before the tensors.

## Generator invocation

The fixture's `regeneratePyTorchSnapshots()` method owns the exact generator
selection. For Qwen3.5 MoE its equivalent command is:

```bash
python3 python/reference/generate_qwen35_moe_pipeline_snapshots.py \
  --model /absolute/path/model.gguf \
  --prompt 'exact prompt text' \
  --output /absolute/path/reference-pack \
  --decode-steps 3
```

The generator also exposes focused modes such as `--metadata-only`,
`--decode-snapshots-only`, and `--mtp-sidecar-snapshots`; read its current
`argparse` definition before use. Focused modes are diagnostic or additive and
must still leave a complete pack for production parity.

CTest commonly constrains OpenMP for native execution. The regeneration wrapper
unsets inherited `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
and affinity variables before invoking PyTorch so large CPU reference work can
use the host economically.

## Safely changing the oracle

When adding a model or checkpoint:

1. Implement or extend a registered reference model and exact GGUF transforms.
2. Define checkpoint names at mathematically equivalent boundaries, including
   router and MTP sidecar stages where applicable.
3. Bump or extend the schema when interpretation or required files change.
4. Update metadata validation and its unit tests before accepting new packs.
5. Generate from a staged real GGUF and inspect tensor shapes, counts, finite
   values, and hook cardinality.
6. Run the native affected cell and use the canonical CSVs to prove equivalence.
7. Run the full globally budgeted campaign.

Do not make the oracle imitate a backend-specific kernel order merely to hide a
defect. If quantized production arithmetic has a justified tolerance, keep the
independent CPU/FP32 reference and express the policy in the typed native
threshold configuration with diagnostic evidence.

Useful device-free checks:

```bash
python3 -m pytest -q \
  python/reference/tests/test_snapshot_metadata.py \
  python/reference/tests/test_dequantize.py \
  python/reference/tests/test_transformers_api_compat.py
```

The similarly named files under `python/reference/loaders/test_*.py` are
manual real-GGUF CLI probes, not pytest suites. Invoke one with an explicit
model when diagnosing parser, dequantization, or full-loader behavior, for
example:

```bash
python3 -m python.reference.loaders.test_parser /absolute/path/model.gguf
```
