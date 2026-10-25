/**
 * @file SharedReadoutLifecycle.cpp
 * @brief Complete two-round timing including deferred raw-leaf key projections.
 *
 * Each transaction verifies two fixed stars, accepts the first leaf then the
 * last, and publishes a terminal dense state. The candidate pays for both raw
 * update resolutions and all adjusted-query preparation. Controls are the same
 * new readout with dense snapshots and the previous exact compact prototype.
 * No allocation, buffer rebinding, or input reset occurs in the timed region.
 */
#include <chrono>
#include <cstring>

extern "C" int gdn_shared_group(int, int, int, int,
    const float *, const float *, const float *, const float *, const float *, const float *,
    float *, float *, float *, float *);
extern "C" int gdn_shared_resolve(int, int, const float *, const float *, const float *, float, float, float *);
extern "C" int gdn_leaf_exact_lifecycle(int, int, int, int, int,
    const float *, const float *, const float *, const float *, const float *, const float *,
    float *, float *, float *, float *, float *, double *);

namespace
{
    /** @brief Explicit, independently checked algorithms; never recovery modes. */
    enum class Mode { SharedLeaves = 0, SharedSnapshots = 1, LegacyCompact = 2 };
}

/**
 * @brief Time a complete repeated two-round transaction against matched inputs.
 * @param mode SharedLeaves=0, SharedSnapshots=1, LegacyCompact=2.
 * @param iterations Positive repetitions of identical complete transactions.
 * @param rows Live siblings per round in [1,4096].
 * @param dk Key width in [1,512].
 * @param dv Value width, a positive multiple of eight through 512.
 * @param state Immutable initial state [dk,dv].
 * @param q Prepared queries [2,rows,dk].
 * @param k Prepared keys [2,rows,dk].
 * @param v Values [2,rows,dv].
 * @param decay Decay gates [2,rows].
 * @param beta Write gates [2,rows].
 * @param output Both rounds' readouts [2,rows,dv].
 * @param corrections Legacy compact workspace [2,rows,dv].
 * @param snapshots Dense-control workspace [2,rows,dk,dv].
 * @param resolved Accepted first-round state [dk,dv].
 * @param terminal Final dense state [dk,dv].
 * @param adjusted Prepared-query workspace [2,rows,dk].
 * @param eta Readout-coefficient workspace [2,rows].
 * @param nanoseconds Mean duration of a complete transaction.
 * @return 1 on success, -1 for an invalid contract or failed native operation.
 */
extern "C" int gdn_shared_lifecycle(int mode, int iterations, int rows, int dk, int dv,
    const float *state, const float *q, const float *k, const float *v,
    const float *decay, const float *beta, float *output, float *corrections,
    float *snapshots, float *resolved, float *terminal, float *adjusted, float *eta, double *nanoseconds)
{
    if (iterations <= 0 || rows <= 0 || rows > 4096 || dk <= 0 || dk > 512 ||
        dv <= 0 || dv > 512 || dv%8 || !state || !q || !k || !v || !decay || !beta ||
        !output || !corrections || !snapshots || !resolved || !terminal || !adjusted || !eta || !nanoseconds)
        return -1;
    if (mode == static_cast<int>(Mode::LegacyCompact))
        return gdn_leaf_exact_lifecycle(0, iterations, rows, dk, dv, state, q, k, v, decay, beta,
            output, corrections, snapshots, resolved, terminal, nanoseconds);
    if (mode != static_cast<int>(Mode::SharedLeaves) && mode != static_cast<int>(Mode::SharedSnapshots))
        return -1;
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        if (gdn_shared_group(mode, rows, dk, dv, state, q, k, v, decay, beta,
            adjusted, eta, output, snapshots) != 1) return -1;
        const float *parent = snapshots;
        if (mode == static_cast<int>(Mode::SharedLeaves))
        {
            if (gdn_shared_resolve(dk, dv, state, k, v, decay[0], beta[0], resolved) != 1) return -1;
            parent = resolved;
        }
        if (gdn_shared_group(mode, rows, dk, dv, parent, q + rows*dk, k + rows*dk, v + rows*dv,
            decay + rows, beta + rows, adjusted + rows*dk, eta + rows, output + rows*dv,
            snapshots + rows*dk*dv) != 1) return -1;
        const int selected = 2*rows - 1;
        if (mode == static_cast<int>(Mode::SharedLeaves))
        {
            if (gdn_shared_resolve(dk, dv, parent, k + selected*dk, v + selected*dv,
                decay[selected], beta[selected], terminal) != 1) return -1;
        }
        else
            std::memcpy(terminal, snapshots + selected*dk*dv, static_cast<size_t>(dk)*dv*sizeof(float));
    }
    *nanoseconds = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count()/iterations;
    return 1;
}
