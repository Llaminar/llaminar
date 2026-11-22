# QFN CUDA decode placement and peer transfers

The 2×CUDA/2×CPU Qwen3.8-Flash-Next UD-Q4_K_XL configuration runs on two
RTX 3090s in the original CUDA rank and two physical CPU sockets. Both GPUs
reported directed peer reads and writes over the PCIe PXB link. CUDA reported
no native peer atomics in either direction.

## Routed expert placement

The same CUDA Release binary ran both placements with a recorded 595-token
prompt, greedy sampling, a 4,096-token context, the Q8_0 predictor at fixed
draft depth seven, an empty prefix cache, and residency maintenance disabled.
The dense tensor-parallel mapping and the routed GPU budget of 112 experts
per layer were identical. Whole-expert apportionment used its native
collective continuation transport. Tensor sharding used the mapped sparse
continuation transport.

| Placement | Median decode tokens/s | Draft steps | Accepted | Rejected | Verifier runs | Verifier rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tensor-sharded GPU and CPU tiers | 44.12 | 399 | 198 | 44 | 57 | 456 |
| Apportioned GPU and CPU tiers | 51.82 | 364 | 204 | 31 | 52 | 416 |

Each arm repeated its 256-token history across three measurements. The first
different token was token 10, with IDs 8385 and 4087, respectively. A second
experiment generated 64 tokens in ordinary decode and at fixed draft depth
seven. Both placements matched their own ordinary token history byte for byte.
The two ordinary histories first differed at the same token. The throughput
figures above describe different generated histories and verifier workloads.
The CUDA fusion measurements use the tensor-sharded reference, with 320 local
routed columns and 320 local shared columns.

A captured arithmetic witness used the original CUDA Down entry and identical
packed operands. It compared a full 640-column projection with the ordered FP32
sum of its two 320-column fragments. Both GPUs reproduced different FP32 output
bytes for Q5_1 and Q8_0 in all eight input states. The largest differences were
on the order of 10⁻⁵. The witness establishes that the source-aligned split
changes the reduction tree. The model comparison also changes the continuation
transport and prefill implementation.

Increasing the tensor-sharded GPU expert budget from 112 to 144 produced all
256 original tokens and identical MTP counts above. The first pair measured
44.12 and 45.16 tokens/s at 112 and 144, respectively. The reverse order
measured 43.32 and 44.38 tokens/s at 144 and 112. The retained prepared weights
increased by 686,387,200 bytes per GPU under the existing memory admission,
with identical retained workspace bytes. The fusion control retains 112
pending a repeatable gain.

## Inter-card decode payloads

The native Release benchmark captures exact cross-device dependencies and
prebinds the two input rows. It checks the receiver's complete bytes after
three independent input resets and reports native CUDA-event timings. The
width sweep used both directions, four block widths, and 43 samples per cell.
The installed NCCL provider reported version 2.28.9. The fastest measured
rooted publication uses 32 threads at 10 KiB and 256 threads at 100 KiB.

| FP32 payload | CUDA operation | Median range, µs | NCCL Reduce median range, µs |
| --- | --- | ---: | ---: |
| 10 KiB | Source reads both inputs and publishes the sum directly to receiver VRAM | 3.62–3.65 | 9.86–10.08 |
| 100 KiB | Root reads the peer input and publishes the sum locally | 10.75 | 24.83–30.69 |
| 400 KiB | Root reads the peer input and publishes the sum locally | 33.50–33.82 | 56.51–60.42 |

The 10 KiB direct copy took 2.27 µs and the peer DMA copy took 2.85–3.05 µs.
An additional protocol used aligned mapped-host readiness words with native
release/acquire and wrote the payload to receiver VRAM. It passed replay in
both directions and took 10.72–10.78 µs at 10 KiB and 20.74–20.96 µs at
100 KiB with the best tested widths. The event-ordered prototype is faster
for the measured cases. These tests start with ready inputs. Integrating the
event path into a production layer requires a captured dependency from both
original participants and retention through the last peer read or write.

The explicit Release target and reproduction commands are in the CUDA
roofline benchmark documentation.
[CUDA roofline and peer timing](../../../../tests/v2/performance/kernels/cuda/roofline/README.md)
