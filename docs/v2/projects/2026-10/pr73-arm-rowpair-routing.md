# ARM ordered row decomposition — implementation checkpoint

UNVERIFIED: committed before builds/tests by owner instruction. No acceptance or
stack qualification is claimed. Performance optimization is deferred; retained
economy16 and previous native runs must not be overwritten.

`kNEONRowDecompositions` is the exact ordered-only source-key inventory (13
keys): ASIMD 3-row IQ3/source-IQ4 and 4-row IQ3/IQ1/Q4_K;
SDOT 3-row IQ3/IQ1/source-IQ4/Q4_K and 4-row IQ3/IQ1/Q4_K;
I8MM 4-row IQ3. The first physical program consumes rows [0,2), the
second [2,Rows). All K geometry, column chunk and exact spans are retained.
ASIMD/SDOT retain their families; the I8MM IQ3 source uses SDOT pairs because
the existing I8MM two-row IQ3 key fails its current stack gate.

Expected retired physical keys by envelope: NEON 5, DOTPROD 12, I8MM 13.
Existing smaller leaf keys are reused; no larger source program is instantiated
by the production dispatch branches. The test-only wide oracle deliberately
instantiates originals in the numerical test executable, not the probe.

Pending independent review BEFORE changing effective key/inventory handling:

- The register witness currently expects a Cartesian row/family/encoding set.
  It must instead expect the union of actually dispatched leaves under this
  exact typed routing policy, reject retired wide symbols, and audit dispatch
  closure to the precise smaller-row targets. Do not weaken stack accounting,
  committed budgets, baselines, or role classification.
- GCC attribute key lists and current role/count inventory must remove retired
  keys only after that review. Smaller keys retain their existing identities
  and budgets. None of those proof artifacts is modified in this checkpoint.
- Qualify each replacement leaf against CURRENT source/envelope measurements,
  not merely absence from a failure list. Original failures remaining at M=1
  and I8MM M=2 are explicitly outside this implementation's claim.
- Run focused original-wide versus decomposed numerical test, full native
  all-format and NaN-priority contracts, and all three actual-dispatch stack
  gates. No tests have yet run. The focused regression is explicitly registered
  in ProductionTestPreflight; it covers both policies, ragged N, partition
  boundaries, exact row offsets/guards and repeated dirty-buffer execution.

Partition-carry/shared-decode redesign is a separate future optimization, not
part of this implementation. This policy accepts payload rereads; it never
executes serial-row replay or allocates unused row capacity.
