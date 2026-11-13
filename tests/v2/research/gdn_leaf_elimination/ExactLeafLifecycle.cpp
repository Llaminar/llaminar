/**
 * @file ExactLeafLifecycle.cpp
 * @brief Allocation-free two-round timing boundary for the compact leaf trial.
 *
 * Both candidates start from the same immutable state, verify two fixed star
 * trees, accept row zero after the first and the final row after the second,
 * and publish the exact terminal dense state. The compact candidate must pay
 * for fused old-pending resolution and terminal materialization. The control
 * uses the identical tiled arithmetic with dense state publication. This is
 * an isolated synthetic experiment, never a production throughput certificate.
 */

#include <chrono>
#include <cstring>

extern "C" int gdn_leaf_exact_outputs(int, int, int, const float *, const float *, const float *,
    const float *, const float *, const float *, float *, float *);
extern "C" int gdn_leaf_exact_snapshots(int, int, int, const float *, const float *, const float *,
    const float *, const float *, const float *, float *, float *, float *);
extern "C" int gdn_leaf_exact_after_pending(int, int, int, const float *, const float *, const float *,
    const float *, const float *, const float *, float *, float *, float, const float *, const float *, float *);
extern "C" int gdn_leaf_exact_materialize(int, int, const float *, float, const float *, const float *, float *);

namespace
{
    /** @brief Public integer selectors have exactly two admitted meanings. */
    enum class Lifecycle { Compact = 0, DenseControl = 1 };

    /**
     * @brief Repeat a complete fixed two-round lifecycle with retained buffers.
     * @tparam Mode Explicit publication policy, selected before timing starts.
     * @return 1 after publishing duration and terminal state; -1 on a kernel error.
     * @see gdn_leaf_exact_lifecycle for all operand and destination contracts.
     */
    template<Lifecycle Mode>
    int measure(int iterations, int rows, int dk, int dv, const float *state,
        const float *q, const float *k, const float *v, const float *decay, const float *beta,
        float *output, float *corrections, float *snapshots, float *resolved, float *terminal,
        double *nanoseconds)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            if constexpr (Mode == Lifecycle::Compact)
            {
                if (gdn_leaf_exact_outputs(rows, dk, dv, state, q, k, v, decay, beta, output, corrections) != 1)
                    return -1;
                if (gdn_leaf_exact_after_pending(rows, dk, dv, state,
                    q + rows * dk, k + rows * dk, v + rows * dv, decay + rows, beta + rows,
                    output + rows * dv, corrections + rows * dv, decay[0], k, corrections, resolved) != 1)
                    return -1;
                const int accepted = 2 * rows - 1;
                if (gdn_leaf_exact_materialize(dk, dv, resolved, decay[accepted],
                    k + accepted * dk, corrections + accepted * dv, terminal) != 1)
                    return -1;
            }
            else
            {
                if (gdn_leaf_exact_snapshots(rows, dk, dv, state, q, k, v, decay, beta,
                    output, corrections, snapshots) != 1)
                    return -1;
                if (gdn_leaf_exact_snapshots(rows, dk, dv, snapshots,
                    q + rows * dk, k + rows * dk, v + rows * dv, decay + rows, beta + rows,
                    output + rows * dv, corrections + rows * dv, snapshots + rows * dk * dv) != 1)
                    return -1;
                std::memcpy(terminal, snapshots + (2 * rows - 1) * dk * dv,
                            static_cast<size_t>(dk) * dv * sizeof(float));
            }
        }
        const auto stop = std::chrono::steady_clock::now();
        *nanoseconds = std::chrono::duration<double, std::nano>(stop - start).count() / iterations;
        return 1;
    }
}

/**
 * @brief Time a complete matched lifecycle without Python call overhead.
 * @param mode 0 for compact updates, 1 for the dense-snapshot control.
 * @param iterations Positive repeat count; every iteration has identical inputs.
 * @param rows Positive live leaf count per round.
 * @param dk Key width in [1,512].
 * @param dv Value width, a positive multiple of eight through 512.
 * @param state Immutable initial state, [dk,dv].
 * @param q Prepared queries for two rounds, [2,rows,dk].
 * @param k Keys for two rounds, [2,rows,dk].
 * @param v Values for two rounds, [2,rows,dv].
 * @param decay FP32 decay gates, [2,rows].
 * @param beta FP32 write gates, [2,rows].
 * @param output Both rounds' published readouts, [2,rows,dv].
 * @param corrections Both rounds' exact correction records, [2,rows,dv].
 * @param snapshots Control-only storage [2,rows,dk,dv], untouched in compact mode.
 * @param resolved Compact mode's retained parent of round two, [dk,dv].
 * @param terminal Dense terminal state after the second accepted leaf, [dk,dv].
 * @param nanoseconds Mean wall nanoseconds per complete two-round transaction.
 * @return 1 on success, -1 for invalid input or a native contract violation.
 */
extern "C" int gdn_leaf_exact_lifecycle(int mode, int iterations, int rows, int dk, int dv,
    const float *state, const float *q, const float *k, const float *v,
    const float *decay, const float *beta, float *output, float *corrections,
    float *snapshots, float *resolved, float *terminal, double *nanoseconds)
{
    if (iterations <= 0 || rows <= 0 || rows > 4096 || dk <= 0 || dk > 512 ||
        dv <= 0 || dv > 512 || dv % 8 || !nanoseconds || !state || !q || !k || !v ||
        !decay || !beta || !output || !corrections || !snapshots || !resolved || !terminal)
        return -1;
    if (mode == static_cast<int>(Lifecycle::Compact))
        return measure<Lifecycle::Compact>(iterations, rows, dk, dv, state, q, k, v, decay, beta,
            output, corrections, snapshots, resolved, terminal, nanoseconds);
    if (mode == static_cast<int>(Lifecycle::DenseControl))
        return measure<Lifecycle::DenseControl>(iterations, rows, dk, dv, state, q, k, v, decay, beta,
            output, corrections, snapshots, resolved, terminal, nanoseconds);
    return -1;
}
