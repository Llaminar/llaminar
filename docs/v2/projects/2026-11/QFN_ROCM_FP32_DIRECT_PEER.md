# Opt-in ROCm FP32 direct peer reduction

`LLAMINAR_RCCL_RANK_ORDERED_PUSH=1` selects the direct-peer experiment before original RCCL coordinator creation. The setting is process-lifetime and defaults off. Start a fresh process for each selection. Continue to specify the same model, original tensor-parallel placement, request shape, MTP mode and precision. The switch does not change model or verifier arithmetic.

The experiment requires an authenticated complete directed P2P ROCm clique within one MPI rank, the native communicator's abort authority and the full admitted resident grid on each participant. The installation candidate admits native gfx906. Successful gfx1100 compilation establishes code generation, while runtime visibility, scheduling and economics remain unqualified on that target. The original RCCL communicator and model-owned PMA attachment outlive every experimental graph reader. Original LocalTP retirement of the peer attachment releases the model charges before parking the RCCL communicator. An explicitly selected incompatible cohort fails admission.

The existing canonical FP32 rank-order stage uses direct peer for admitted captured storage rows 1 through 63 and a positive row width with at most 262,144 elements. Capacity determines launch geometry. The device-owned equal row prefix determines useful wire bytes. Empty prefixes carry protocol progress only. Larger captured footprints retain the canonical gather/fold regime. For the ordinary attention graph, local TP degree above two selects canonical rank order during decode. The F-sharded MoE final sum declares canonical FP32 separately, including grouped and eligible small predictor or prefill shapes. Unrelated RCCL operations and CUDA/CPU communication remain under their original stages.

The communicator's original logical member order determines the FP32 fold. Peer endpoints use physical ROCm ordinals solely for ownership and placement. In round one each peer publishes its own FP32 block to the receivers' admitted uncached inboxes. The acquire joins precede ascending left-fold additions and local publication. Ping-pong epochs require ordered collective submissions on the original participant graph. Split streams are safe only where the existing graph orders their reuse through its explicit event joins.

## Qualification

1. Run the installed device-free ProductionTestPreflight layout, policy, and PMA quote regressions. Verify the exact Release runtime, original MPI ranks, complete directed physical peer scope, configured RCCL provider and actually loaded HIP code.
2. Run the native captured Replay, MixedPrefill, Abort and OwnerReplacement cases. Check dirty zero/partial/full transitions, candidate/control byte equality, actual useful and protocol byte receipts, missing-peer native abort, and physical retirement.
3. Run separate native production capture and unprofiled model timing windows. For the capture use `LLAMINAR_GPU_GRAPH_KERNEL_INVENTORY=1`, a unique `LLAMINAR_PERF_STATS_JSON` path and filters `gpu_graph_inventory,tp_allreduce.canonical_rank_order_reductions`. Require resolved `peerPushFP32Kernel` on each admitted device and matching `transport=native_peer_push` and `arithmetic=device_rank_order` capture counters. The off arm must show the original gather/fold transport for the same selected stage set and rank order. Capture inventory counts are setup records.
4. Compare byte-identical frozen histories and repeatable generated tokens in ordinary Off, Fixed and Dynamic MTP with partial, empty and reset transactions. Include real ordered sidebands, HYBRID PP/TP coexistence and legal stream-overlap variants for any contemplated default scope. Native P2 and P4 tests qualify those measured cohorts, while other peer degrees retain their structural host coverage until native qualification.
5. Prove repeated full graph owner retirement and re-creation within one process, including a new physical memory authority and a pooled communicator. The owner-replacement regression checks that an active foreign PMA is rejected, both admitted model graphs replay exactly, each model PMA returns to zero, and the pooled communicator and abort identities remain identical. Keep the option off by default until the broader production scopes above have native proof.


## Reproducing the native and full-model measurements

Use a Release build with ROCm and RCCL, the configured MPI session launcher,
and one original MPI rank per socket. The timing fixture discovers four
rank-local directed peers. The owner-replacement fixture discovers two.

```bash
cmake --build "$BUILD_DIR" --parallel --target v2_integration_rocm_peer_push v2_test_peer_push_reduction v2_test_memory_planner
ctest --test-dir "$BUILD_DIR" -C Release --no-tests=error --output-on-failure -R '^V2_(Unit_PeerPushReduction|Integration_(PeerPushReductionLayout|RCCLRankOrderedAdmission|ROCmPeerPushCaptured(Replay|MixedPrefill|Abort|OwnerReplacement)))$'
ctest --test-dir "$BUILD_DIR" -C Release --no-tests=error -V -R '^V2_Perf_ROCmPeerPushCapturedTiming$'
```

On four same-switch MI50s, 97 reductions of 2,560 FP32 elements took
26.81 µs each with gather/fold and 12.15 µs with direct push. The isolated
sweep used an A/B/B/A run order and 31 repeats per arm.

The Qwen3.8-Flash-Next UD-Q4_K_XL model comparison used 512-token review
and 2,048-token code prompts, an empty prefix cache, one warmup and three
256-token generations per prompt. Both arms used the same Release binaries,
rank-owned ROCm TP4, tensor-sharded experts, RAM ngrams and retained M1
HC/FFN. The transport selector was the only A/B change. Ordinary decode
improved from 68.28 to 78.30 tok/s on review and from 64.68 to 74.05 on
code. The final owner-lifetime build repeated 78.23 and 74.04 tok/s;
all six histories matched the earlier peer build. The dashboard records the
separate fixed-depth and dynamic MTP runs.
