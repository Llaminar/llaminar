# Idea 13 — Fewer OpenMP regions per decode step

Status: **Partial**. `27585692d` installs a bounded libgomp idle spin
(`GOMP_SPINCOUNT=3000000`, `MPIBootstrap::configureOpenMPWaitPolicy`,
regression `V2_Integration_OpenMPIdleSpinPolicy`). `408ff90e5` removes 15 of
16 barriers from the CPU MoE fused gate/up bundle, now **verified end to end**
(2026-11-08, GB10, 35B): MTP dyn 17.45→27.19±0.18 (+56%), no-MTP 13.27→20.48
(+54%), prefill 37.1–37.8 (+65%); C true no-MTP stage timings `MOE_EXPERT_FFN`
274 µs/call and router bundle 176 µs/call (16 projections, K=2048); MTP gain
over no-MTP at C +32.8%. The bundle part of this idea and of idea 16 is
closed. Reducing the region and barrier count **further is still open** — the
remaining structure (other workshares/regions per step, experts/shared/GDN
pooling candidates, retained-team-per-layer) is unmeasured.

## Finding

A CPU decode step runs many short parallel regions or workshares (per layer:
norms, projections, router, expert bundles, shared expert, GDN pieces,
residual) separated by short serial work. With libgomp's default 300k-spin
budget, workers futex-sleep inside those gaps, and every region pays a full
team wake-up. On GB10's mixed cores this made the 35B grouped MTP verifier
cost about as much per row as serial decode:

- MTP dynamic: 13.23 -> 18.73 tok/s with spin 3M;
- MTP off: 13.68 -> 14.34 tok/s.

Per-call kernel times barely moved (router 668 -> 651 us, experts 803 -> 774
us), so the gain is in the gaps between regions, not inside kernels.

The spin budget hides wake-ups but still burns cycles at every barrier. The
structural fix is fewer barriers.

## Ideas

1. **Pool independent members into one workshare** wherever a stage issues
   several independent projections back to back. `408ff90e5` did this for the
   fused NEON bundle, and the same pattern applies to:
   - routed experts (gate/up then down per expert),
   - the shared expert,
   - the GDN projection pieces.

   Tasks must keep whole output tiles with fixed full-K arithmetic, so the
   change stays byte-neutral.
2. **One retained team per layer or per step** (`OMP_WORKSHARE_REGION`
   already supports running inside an existing team). Measure the barrier
   count per decode step before and after with a PerfStats counter, not by
   inference.
3. **Re-tune the spin budget per host class** only with the sweep and
   idle-cost evidence (see `00-measured-baseline.md`). Never use an unbounded
   spin: 19 of 20 GB10 cores stay busy on an idle server.
4. **x86 confirmation.** The same libgomp policy applies on x86 GCC builds.
   The effect there has not been measured.

## Exactness / risk

- Scheduling only; arithmetic per output tile unchanged. Byte-authenticate
  every change with the existing serial-row oracles.

## Validation

- Per-stage CPU timing (`LLAMINAR_PERF_STATS_CPU_STAGE_TIMING=1`) for 35B MoE
  and 27B dense, MTP off and dynamic, before and after.
- `Perf__CPUFusedVerifier ManyNarrowMembersBundleVersusStandalone` for
  bundle-shaped work.
