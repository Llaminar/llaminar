# Native W4A4/H reference

`w4a4.py` owns the independent CPU/PyTorch arithmetic reference for the pinned
ruquant export. This is **not** a dequantized-weight approximation or a statement
that Llaminar's production backends are qualified.

## Identity

- Model: `raspbfox/Qwen3.8-27B-W4A4-GGUF`, revision
  `174067dc0522d3c69912487e799b4e9541cb296c`.
- Target file: `Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf`, 17,190,149,440 bytes,
  advertised SHA-256
  `b1015b212068a2838f23a68985065352e337d9e4e0f40abab3832e610a6a5ef8`.
  Acquisition must verify the complete file independently before real-model use.
- Native arithmetic: `skyne98/mx-llama.cpp`, revision
  `bfc4c5dc3c96df652f42924370ef49b728d585ad`.
- Exporter: `skyne98/ruquant`, revision
  `7851ba11eddf3de44adc9dba0ccb24472b9ba30e`.
- Policy: `ruquant-w4a4-g64-shift4-v1`.
- HF/PyTorch environment: [`w4a4-requirements.txt`](w4a4-requirements.txt).
  The loader checks the installed Transformers VCS receipt, Qwen35 implementation
  SHA-256 and PyTorch version. An upstream release with a similar version/name
  does not substitute for the pinned fork's dense GGUF head-order contract.

Install the reference recipe in an owned environment rather than changing a
shared tuning environment. Arithmetic-only tests need CPU PyTorch/NumPy; the
HF graph contract additionally requires that exact fork.

## Numerical contract

A source block stores 256 signed int4 weights in 144 bytes: one FP32 row unit,
a descriptor carrying either the whole-row code sum or integer weight-group
multipliers, the stored row/rotation length, the clipping scalar's bits, and
128 packed-code bytes. The decoder retains integer codes, multipliers and FP32
units; it rejects malformed geometry, inconsistent repeated metadata,
nonfinite/negative scales and an unproven int32 accumulation width.

W4A4H first transforms each declared contiguous block using the pinned ordered
Paley/Sylvester transform. The stored length is authoritative: the target's
attention output uses 1536 and FFN down uses 4352, but neither four blocks nor
these two lengths define the general transform contract. The converter already
folds the main hidden-space R1 into the exported weights/embedding; the reference
does not apply that offline rotation a second time.

A4 uses group-64 ranges including zero, a **whole-token** maximum range, a shared
FP32 unit, integer group multipliers, ties-to-even zero points/codes and int4
clamping. Clipping occurs before range selection. The all-zero row policy is
explicit. Splitting K and independently choosing each partition's unit is not
the same operation.

For each group, the signed dot is corrected by `(8-z)*weight_code_sum`, multiplied
by both integer multipliers, then arithmetically shifted by four **before** the
int32 group sum. The epilogue rounds in this order:

```text
((float32(total) * 16) * activation_unit) * weight_unit
```

The reference uses batched FP32 BLAS only for small, exactly representable
integer group dots. Correction, multipliers and shifts stay integer. Hadamard
Paley sums have explicit source order. Autocast and scale/device conversion are
rejected; no reduced-precision arithmetic is silently selected.

## HF binding and MTP

`GGUFWeightMode.NATIVE_OPERANDS` yields checked operands, not floating tensors.
The dense Qwen35 loader replaces each corresponding HF linear module with
`W4A4Linear` and retires the old floating parameter. Other source formats keep
their declared reference operations. Ordinary state-dictionary mode rejects
W4A4/H rather than pretending that a floating weight matrix is equivalent.

The export's main graph is rotated while the converter copies the trailing
nextn block. Tensor presence alone does not authenticate that interface. Native
W4A4 MTP reference execution currently fails with a precise basis-contract error;
it is not silently disabled or certified by a main-model test. A validated
embedding/hidden/head basis contract is required before admission. DFlash2 is
outside this implementation.

## Fast gates and recorded evidence

```bash
python -m pytest python/reference/tests/test_w4a4.py \
  python/reference/tests/test_w4a4_hf.py -q
```

These files join `V2_Unit_Python_Reference_All`. Explicit preflight entries are
`V2_Integration_W4A4ReferenceReplay` and
`V2_Integration_W4A4HFReferenceContracts`. The latter uses a small complete hybrid
HF fixture to check native module binding, prefill, cached decode, reset and
refusal of unqualified MTP; it is not a real-model accuracy certificate.

[`native_trace_v1.json`](../tests/fixtures/w4a4/native_trace_v1.json) records complete
input bytes, source weights, ordered transform outputs, packed A4 operands and
linear outputs from the pinned upstream's actual compiled host bodies. It covers
zero/skewed rows, rounding, negative shifts, all source multiplier encodings and
the target transform geometries. Tests replay recorded inputs through this
independent reference, including large-to-small-to-large batches, and reject
incomplete/incompatible traces. They do not regenerate expectations.

To author a **new, reviewed** trace, use an unmodified pinned checkout:

```bash
cmake -S python/reference/tests/fixtures/w4a4/oracle -B /owned/oracle-build -G Ninja \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" \
  -DCMAKE_BUILD_TYPE=Release -DORACLE_SOURCE=/owned/mx-llama.cpp \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=OFF \
  -DGGML_CPU_REPACK=OFF -DGGML_OPENMP=OFF
cmake --build /owned/oracle-build --parallel
python python/reference/tests/fixtures/w4a4/record_native.py \
  --source /owned/mx-llama.cpp --build /owned/oracle-build \
  --output /owned/new-reviewed-trace.json
```

The recorder refuses an existing destination, altered source or `RQ_` overrides.
The trace retains source/DSO digests and the two arithmetic translation units'
compile commands. Ordinary tests require neither that checkout nor a compiler,
network, model download or accelerator. Larger model/campaign corpora belong in
`Llaminar/corpora`, not beside these small device-free fixtures.
