#!/usr/bin/env python3
"""
Quartic-oscillator Schrödinger vs Hamilton–Jacobi action-density stress test
=============================================================================

Tests the claim from Lohmiller & Slotine (2026), "On computing quantum waves
exactly from classical action" (arXiv:2405.06328, Proc. R. Soc. A 482:20250413),
that the exact Schrödinger wavefunction can be reconstructed from classical
Hamilton–Jacobi trajectories, action accumulation, and a classical density.

The stress test uses a 1-D quartic oscillator::

    V(x) = ½x² + λx⁴   (nondimensional units: m = ℏ = ω = 1)

which is the simplest smooth non-quadratic system not covered by the
exactly-quadratic cases in the paper.  We compute independently:

1. A direct numerical Schrödinger solution (split-operator Fourier method).
2. A classical multibranch Hamilton–Jacobi / action-density reconstruction.

Then we compare them via phase-aligned L₂ error and fidelity and report the
results objectively — this is a numerical stress test, not a final verdict.

QUANTUM SOLVER
--------------
Split-operator Fourier method (Strang splitting, 2nd order in dt)::

    ψ(t+dt) ≈ exp(-i V dt/2) · IFFT[ exp(-i k²dt/2) · FFT[ exp(-i V dt/2) ψ(t) ]]

CLASSICAL HJ RECONSTRUCTION
-----------------------------
For each label q on a dense initial grid:

1. Evolve trajectory::

       dx/dt = p,   dp/dt = -V'(x)

2. Accumulate the classical action (Lagrangian integral)::

       dS/dt = p²/2 - V(x)

3. Evolve the full 1-D monodromy / variational equations::

       dA/dt = C,   dB/dt = D
       dC/dt = -V''(x) A,   dD/dt = -V''(x) B

       M = [[A, B], [C, D]]
         = [[∂x(t)/∂q₀, ∂x(t)/∂p₀],
            [∂p(t)/∂q₀, ∂p(t)/∂p₀]]

       The original HJ Jacobian variables are J=A and P=C for this test's
       uniform initial momentum field p₀(q)=const.

4. Track Maslov index μ: increment by 1 each time J passes through zero
   (a caustic, i.e., a fold in the Lagrangian map).

Reconstruct the wavefunction by finding monotonic branches of the map
q → x(t;q) and summing over branches at each output grid point::

    ψ_HJ(x,t) = Σ_branches sqrt(ρ₀(q_j)/|J_j|) · exp(i S_j/ℏ - i π μ_j/2)

METRICS
-------
Phase-aligned L₂ error::

    ε(t) = min_θ ||ψ_Sch - e^{iθ} ψ_HJ||₂

Fidelity::

    F(t) = |⟨ψ_Sch|ψ_HJ⟩|²

If ε(t) ≈ machine precision, the construction is exact.
If ε(t) grows after caustics / nonlinear distortion, it is semiclassical only.

USAGE
-----
Quick smoke test (small grid, short time)::

    python experiments/quantum_hj_quartic.py --N 128 --T 0.5 --dt 0.01

Standard run (saves CSV and plots to /tmp/qhj)::

    python experiments/quantum_hj_quartic.py --out /tmp/qhj

Full run with all parameters::

    python experiments/quantum_hj_quartic.py \\
        --N 512 --L 8.0 --T 3.0 --dt 0.002 --lambda_ 0.1 \\
        --x0 0.0 --p0 1.0 --sigma 0.5 --N-traj 2000 \\
        --out results/quartic_hj

OUTPUT
------
``diagnostics.csv`` — columns: ``t,L2_error,fidelity``

Optional PNG plots (if matplotlib is available):
``density_snapshots.png`` — |ψ|² comparison at selected times
``metrics.png`` — L₂ error and fidelity vs time

REFERENCES
----------
- Lohmiller & Slotine (2026), arXiv:2405.06328 / Proc. R. Soc. A 482:20250413
- Van Vleck (1928), Proc. Nat. Acad. Sci. 14, 178
- Miller (1970), J. Chem. Phys. 53, 3578 (semiclassical IVR)
- Heller (1991), J. Chem. Phys. 94, 2723 (Gaussian wavepackets)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)
SNAPSHOT_RECORD_LIMIT = 200

# ──────────────────────────────────────────────────────────────────────────────
# Potential and its derivatives  (nondimensional: m = ℏ = ω = 1)
# ──────────────────────────────────────────────────────────────────────────────


def potential(x: np.ndarray, lam: float) -> np.ndarray:
    """V(x) = ½x² + λx⁴"""
    return 0.5 * x**2 + lam * x**4


def potential_first_deriv(x: np.ndarray, lam: float) -> np.ndarray:
    """V'(x) = x + 4λx³"""
    return x + 4.0 * lam * x**3


def potential_second_deriv(x: np.ndarray, lam: float) -> np.ndarray:
    """V''(x) = 1 + 12λx²"""
    return 1.0 + 12.0 * lam * x**2


# ──────────────────────────────────────────────────────────────────────────────
# Initial wavepacket
# ──────────────────────────────────────────────────────────────────────────────


def initial_gaussian(
    x: np.ndarray,
    x0: float,
    p0: float,
    sigma: float,
) -> np.ndarray:
    """Coherent-state Gaussian packet (ℏ = 1)::

        ψ₀(x) = (2πσ²)^{-1/4} exp(-(x-x₀)²/(4σ²) + i p₀ x)

    Normalized so that ∫|ψ₀|² dx = 1.
    """
    norm = (2.0 * np.pi * sigma**2) ** (-0.25)
    return norm * np.exp(-(x - x0) ** 2 / (4.0 * sigma**2) + 1j * p0 * x)


def initial_density(q: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    """ρ₀(q) = |ψ₀(q)|² = (2πσ²)^{-1/2} exp(-(q-x₀)²/(2σ²))"""
    return (2.0 * np.pi * sigma**2) ** (-0.5) * np.exp(
        -(q - x0) ** 2 / (2.0 * sigma**2)
    )


def initial_action_deriv(q: np.ndarray, p0: float) -> np.ndarray:
    """S₀'(q) = p₀  (uniform initial momentum for the coherent-state packet)"""
    return np.full_like(q, p0, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Split-operator Schrödinger solver
# ──────────────────────────────────────────────────────────────────────────────


def schrodinger_step(
    psi: np.ndarray,
    V_arr: np.ndarray,
    k2_arr: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance ψ by one time step using 2nd-order Strang splitting.

    The Hamiltonian is H = p²/(2m) + V(x) with m = ℏ = 1, so the kinetic
    operator in k-space is k²/2.  The splitting::

        ψ → exp(-i V dt/2) · IFFT[ exp(-i k²dt/2) · FFT[ exp(-i V dt/2) ψ ]]

    Parameters
    ----------
    psi:
        Current wavefunction on spatial grid, shape (N,).
    V_arr:
        Potential values on spatial grid V(x_j), shape (N,).
    k2_arr:
        Squared wavenumbers k²_j (FFT ordering), shape (N,).
    dt:
        Time step.

    Returns
    -------
    psi_new : complex array of shape (N,)
    """
    # Half-step in position space
    psi = np.exp(-0.5j * V_arr * dt) * psi
    # Full step in momentum space (kinetic = k²/2)
    psi_k = np.fft.fft(psi)
    psi_k = np.exp(-0.5j * k2_arr * dt) * psi_k
    psi = np.fft.ifft(psi_k)
    # Half-step in position space
    psi = np.exp(-0.5j * V_arr * dt) * psi
    return psi


def run_schrodinger(
    psi0: np.ndarray,
    x_grid: np.ndarray,
    lam: float,
    dt: float,
    n_steps: int,
    record_every: int = 1,
) -> tuple[np.ndarray, List[np.ndarray]]:
    """Run the split-operator Schrödinger solver.

    Parameters
    ----------
    psi0:
        Initial wavefunction, shape (N,).
    x_grid:
        Spatial grid, shape (N,), assumed uniform with spacing dx.
    lam:
        Quartic coupling λ.
    dt:
        Time step.
    n_steps:
        Number of time steps.
    record_every:
        Save a snapshot every this many steps (step 0 is always included).

    Returns
    -------
    times : array of shape (n_records,), not including t=0
    psi_list : list of n_records wavefunction arrays, not including psi0
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    # Wavenumber grid in FFT ordering
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    k2_arr = k**2  # kinetic energy = k²/(2m) = k²/2 for m=1

    V_arr = potential(x_grid, lam)

    psi = psi0.copy().astype(complex)
    times: List[float] = []
    psi_list: List[np.ndarray] = []

    for step in range(1, n_steps + 1):
        psi = schrodinger_step(psi, V_arr, k2_arr, dt)
        if step % record_every == 0:
            times.append(step * dt)
            psi_list.append(psi.copy())

    return np.array(times), psi_list


# ──────────────────────────────────────────────────────────────────────────────
# Classical trajectory ODE (vectorized over N_traj trajectories)
# ──────────────────────────────────────────────────────────────────────────────


def traj_rhs(state: np.ndarray, lam: float) -> np.ndarray:
    """Right-hand side of the trajectory ODE system (vectorized).

    State is a flat concatenation over N_traj trajectories::

        state = [x₁..xₙ | p₁..pₙ | S₁..Sₙ | A₁..Aₙ | B₁..Bₙ | C₁..Cₙ | D₁..Dₙ]

    Equations (m = ℏ = 1)::

        dx/dt = p
        dp/dt = -V'(x)
        dS/dt = p²/2 - V(x)   (Lagrangian)
        dA/dt = C
        dB/dt = D
        dC/dt = -V''(x) · A
        dD/dt = -V''(x) · B

    Parameters
    ----------
    state:
        Concatenated state array of length 7·N_traj.
    lam:
        Quartic coupling λ.

    Returns
    -------
        rhs : array of length 7·N_traj
    """
    n = len(state) // 7
    x = state[0:n]
    p = state[n : 2 * n]
    A = state[3 * n : 4 * n]
    B = state[4 * n : 5 * n]
    C = state[5 * n : 6 * n]

    Vp = potential_first_deriv(x, lam)
    Vpp = potential_second_deriv(x, lam)
    V = potential(x, lam)

    return np.concatenate(
        [
            p,               # dx/dt
            -Vp,             # dp/dt
            0.5 * p**2 - V,  # dS/dt  (Lagrangian)
            C,               # dA/dt
            state[6 * n : 7 * n],  # dB/dt = D
            -Vpp * A,        # dC/dt
            -Vpp * B,        # dD/dt
        ]
    )


def run_trajectories(
    q_arr: np.ndarray,
    p0_arr: np.ndarray,
    S0_arr: np.ndarray,
    lam: float,
    dt: float,
    n_steps: int,
    record_every: int = 1,
) -> tuple[np.ndarray, List[dict]]:
    """Run classical trajectories using a vectorized 4th-order Runge–Kutta.

    Also integrates the full 1-D monodromy matrix
    ``M = [[A, B], [C, D]]`` and tracks the Maslov index for each trajectory.
    The legacy HJ Jacobian fields ``J`` and ``P`` are emitted as aliases for
    ``A`` and ``C`` because this experiment uses uniform initial momentum.

    Parameters
    ----------
    q_arr:
        Initial trajectory labels (positions), shape (N_traj,).
    p0_arr:
        Initial momenta p₀(q), shape (N_traj,).
    S0_arr:
        Initial actions S₀(q), shape (N_traj,).
    lam:
        Quartic coupling λ.
    dt:
        Time step.
    n_steps:
        Number of time steps.
    record_every:
        Save a snapshot every this many steps.

    Returns
    -------
    times : array of shape (n_records,)
    snapshots : list of dicts, each with keys:
        ``x``, ``p``, ``S``, ``A``, ``B``, ``C``, ``D``, ``J``, ``P``
        (shape (N_traj,)) and ``mu`` (integer array, shape (N_traj,))
    """
    n = len(q_arr)
    # Full monodromy initial condition is identity.
    A0 = np.ones(n)
    B0 = np.zeros(n)
    C0 = np.zeros(n)
    D0 = np.ones(n)

    state = np.concatenate([q_arr, p0_arr, S0_arr, A0, B0, C0, D0])

    # Maslov index: increment by 1 at each caustic (J=A sign change)
    mu = np.zeros(n, dtype=int)
    prev_sign_J = np.sign(A0)  # initially all +1

    times: List[float] = []
    snapshots: List[dict] = []

    def _rk4(st: np.ndarray) -> np.ndarray:
        k1 = traj_rhs(st, lam)
        k2 = traj_rhs(st + 0.5 * dt * k1, lam)
        k3 = traj_rhs(st + 0.5 * dt * k2, lam)
        k4 = traj_rhs(st + dt * k3, lam)
        return st + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    for step in range(1, n_steps + 1):
        state = _rk4(state)

        # Detect caustic crossings: sign change of J
        J_now = state[3 * n : 4 * n]
        new_sign = np.sign(J_now)
        crossed = (new_sign != 0) & (prev_sign_J != 0) & (new_sign != prev_sign_J)
        mu[crossed] += 1
        prev_sign_J = np.where(new_sign != 0, new_sign, prev_sign_J)

        if step % record_every == 0:
            times.append(step * dt)
            snapshots.append(
                {
                    "x": state[0:n].copy(),
                    "p": state[n : 2 * n].copy(),
                    "S": state[2 * n : 3 * n].copy(),
                    "A": state[3 * n : 4 * n].copy(),
                    "B": state[4 * n : 5 * n].copy(),
                    "C": state[5 * n : 6 * n].copy(),
                    "D": state[6 * n : 7 * n].copy(),
                    "J": state[3 * n : 4 * n].copy(),
                    "P": state[5 * n : 6 * n].copy(),
                    "mu": mu.copy(),
                }
            )

    return np.array(times), snapshots


# ──────────────────────────────────────────────────────────────────────────────
# HJ wavefunction reconstruction
# ──────────────────────────────────────────────────────────────────────────────


def reconstruct_psi_hj(
    x_grid: np.ndarray,
    q_arr: np.ndarray,
    rho0_arr: np.ndarray,
    snap: dict,
    caustic_width: float = 0.0,
    post_smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Reconstruct ψ_HJ on *x_grid* from one trajectory snapshot.

    The reconstruction sums over all monotonic branches of the Lagrangian
    map q → x(t;q).  Each branch contributes::

        sqrt(ρ₀(q_j) / |J_j|) · exp(i S_j - i π μ_j / 2)

    evaluated at each output grid point by inverse interpolation within
    the branch.

    Parameters
    ----------
    x_grid:
        Output spatial grid, shape (N_grid,).
    q_arr:
        Trajectory labels (sorted), shape (N_traj,).
    rho0_arr:
        Initial density ρ₀(q), shape (N_traj,).
    snap:
        Trajectory snapshot dict with keys ``x``, ``S``, ``J``, ``mu``.
    caustic_width:
        Optional finite Jacobian floor.  When positive, the Van Vleck
        denominator uses ``sqrt(J² + caustic_width²)`` instead of ``|J|`` to
        suppress unphysical caustic spikes.
    post_smooth_sigma:
        Optional Gaussian smoothing width in x-units, applied after branch
        summation.  This is a controlled finite-width regularization of the
        ray-density result.

    Returns
    -------
    psi : complex array of shape (N_grid,)
    """
    psi = np.zeros(len(x_grid), dtype=complex)

    x_traj = snap["x"]
    S_traj = snap["S"]
    J_traj = snap["J"]
    mu_traj = snap["mu"].astype(float)

    n_traj = len(q_arr)
    sign_J = np.sign(J_traj)

    # Find indices where J changes sign (branch / caustic boundaries)
    change = (
        (sign_J[:-1] != 0)
        & (sign_J[1:] != 0)
        & (sign_J[:-1] != sign_J[1:])
    )
    split_pts = np.where(change)[0] + 1  # first index of each new branch

    branch_starts = np.concatenate([[0], split_pts])
    branch_ends = np.concatenate([split_pts, [n_traj]])

    for b_start, b_end in zip(branch_starts, branch_ends):
        seg_len = b_end - b_start
        if seg_len < 2:
            continue

        x_b = x_traj[b_start:b_end].copy()
        S_b = S_traj[b_start:b_end].copy()
        absJ_b = np.abs(J_traj[b_start:b_end]).copy()
        rho0_b = rho0_arr[b_start:b_end].copy()
        mu_b = mu_traj[b_start:b_end].copy()

        # Ensure the segment is presented in ascending x order for np.interp
        if x_b[-1] < x_b[0]:
            x_b = x_b[::-1]
            S_b = S_b[::-1]
            absJ_b = absJ_b[::-1]
            rho0_b = rho0_b[::-1]
            mu_b = mu_b[::-1]

        # Enforce strict monotonicity (merge duplicate x values if any)
        _, uid = np.unique(x_b, return_index=True)
        if len(uid) < 2:
            continue
        x_b = x_b[uid]
        S_b = S_b[uid]
        absJ_b = absJ_b[uid]
        rho0_b = rho0_b[uid]
        mu_b = mu_b[uid]

        # Find output grid points within this branch's x-range
        mask = (x_grid >= x_b[0]) & (x_grid <= x_b[-1])
        if not np.any(mask):
            continue

        x_eval = x_grid[mask]

        # Interpolate all quantities to output grid points
        S_at_x = np.interp(x_eval, x_b, S_b)
        absJ_at_x = np.interp(x_eval, x_b, absJ_b)
        rho0_at_x = np.interp(x_eval, x_b, rho0_b)
        # Round Maslov index (interpolation may introduce small non-integers)
        mu_at_x = np.round(np.interp(x_eval, x_b, mu_b)).astype(int)

        # Avoid division by zero near caustics.  caustic_width=0 preserves the
        # raw Van Vleck/Jacobian reconstruction.
        if caustic_width > 0.0:
            safe_J = np.sqrt(absJ_at_x**2 + caustic_width**2)
        else:
            safe_J = np.maximum(absJ_at_x, 1e-30)
        amplitude = np.sqrt(np.maximum(rho0_at_x / safe_J, 0.0))
        phase = S_at_x - 0.5 * np.pi * mu_at_x

        psi[mask] += amplitude * np.exp(1j * phase)

    if post_smooth_sigma > 0.0:
        dx = x_grid[1] - x_grid[0]
        psi = gaussian_smooth_periodic(psi, dx, post_smooth_sigma)

    return psi


def gaussian_smooth_periodic(
    values: np.ndarray,
    dx: float,
    sigma: float,
) -> np.ndarray:
    """Smooth a periodic grid function with a Gaussian kernel via FFT."""
    if sigma <= 0.0:
        return values.copy()
    k = 2.0 * np.pi * np.fft.fftfreq(len(values), d=dx)
    filt = np.exp(-0.5 * (sigma * k) ** 2)
    return np.fft.ifft(np.fft.fft(values) * filt)


def continuous_complex_sqrt(values: np.ndarray) -> np.ndarray:
    """Return sqrt(values) with signs chosen continuously along the array.

    Complex square roots are two-valued.  Semiclassical monodromy prefactors
    use one branch continuously along neighboring trajectory labels; the
    principal branch can inject artificial sign flips that look like spurious
    caustics.  This helper flips each sample's sign when doing so keeps it
    closer to the preceding selected value.
    """
    roots = np.sqrt(values.astype(complex))
    if len(roots) < 2:
        return roots
    for idx in range(1, len(roots)):
        if abs(-roots[idx] - roots[idx - 1]) < abs(roots[idx] - roots[idx - 1]):
            roots[idx] = -roots[idx]
    return roots


def reconstruct_gaussian_ivr(
    x_grid: np.ndarray,
    q_arr: np.ndarray,
    rho0_arr: np.ndarray,
    snap: dict,
    packet_width: float,
    use_hk_prefactor: bool = False,
    thawed_width: bool = False,
    full_hk_prefactor: bool = False,
    chunk_size: int = 256,
) -> np.ndarray:
    """Reconstruct ψ from finite-width trajectory packets.

    This replaces singular ray contributions with Gaussian coherent packets
    centered on each classical trajectory.  With ``use_hk_prefactor=True`` it
    additionally applies a Herman-Kluk-style complex stability prefactor.
    ``full_hk_prefactor=True`` uses the full monodromy matrix when available;
    otherwise the historical diagnostic approximation from ``J`` and ``P`` is
    used.  ``thawed_width=True`` propagates each packet's complex width using
    the same full monodromy matrix, yielding a 1-D thawed-Gaussian comparison.
    """
    width = max(float(packet_width), 1e-12)
    dq_weights = trajectory_quadrature_weights(q_arr)
    weights = np.sqrt(np.maximum(rho0_arr, 0.0) * dq_weights)
    norm = (2.0 * np.pi * width**2) ** (-0.25)
    gamma = 1.0 / (2.0 * width**2)

    x_traj = snap["x"]
    p_traj = snap.get("p", np.zeros_like(x_traj))
    S_traj = snap["S"]
    mu_traj = snap["mu"].astype(float)
    A_traj = snap.get("A", snap.get("J"))
    B_traj = snap.get("B")
    C_traj = snap.get("C", snap.get("P"))
    D_traj = snap.get("D")

    prefactor = np.ones_like(x_traj, dtype=complex)
    if use_hk_prefactor:
        if full_hk_prefactor and B_traj is not None and D_traj is not None:
            # 1-D Herman-Kluk prefactor for frozen coherent states with
            # envelope exp[-γ(x-q)²/2].  The square-root branch is left
            # continuous by NumPy's principal sqrt; Maslov phase from J is
            # still tracked separately for direct comparison with the HJ sum.
            prefactor = continuous_complex_sqrt(
                0.5
                * (
                    A_traj
                    + D_traj
                    - 1.0j * gamma * B_traj
                    + 1.0j * C_traj / gamma
                )
            )
        else:
            # Historical diagnostic approximation used before B and D were
            # tracked.  Kept as ``herman_kluk`` for A/B comparisons.
            prefactor = np.sqrt(0.5 * (A_traj + 1.0j * C_traj / gamma))
        prefactor = np.where(np.isfinite(prefactor), prefactor, 0.0)

    denom_sqrt = None
    evolved_gamma = None
    if thawed_width and B_traj is not None and D_traj is not None:
        denom = A_traj + 1.0j * gamma * B_traj
        denom_sqrt = continuous_complex_sqrt(denom)
        evolved_gamma = (D_traj * gamma - 1.0j * C_traj) / denom

    psi = np.zeros(len(x_grid), dtype=complex)
    for start in range(0, len(q_arr), chunk_size):
        end = min(start + chunk_size, len(q_arr))
        dx_mat = x_grid[None, :] - x_traj[start:end, None]
        if denom_sqrt is not None and evolved_gamma is not None:
            envelope = (
                norm
                / denom_sqrt[start:end, None]
                * np.exp(-0.5 * evolved_gamma[start:end, None] * dx_mat**2)
            )
        else:
            envelope = norm * np.exp(-(dx_mat**2) / (4.0 * width**2))
        phase = (
            S_traj[start:end, None]
            + p_traj[start:end, None] * dx_mat
            - 0.5 * np.pi * mu_traj[start:end, None]
        )
        coeff = weights[start:end, None] * prefactor[start:end, None]
        psi += np.sum(coeff * envelope * np.exp(1j * phase), axis=0)

    return psi


def trajectory_quadrature_weights(q_arr: np.ndarray) -> np.ndarray:
    """Return trapezoidal quadrature weights for trajectory labels."""
    if len(q_arr) == 0:
        return np.array([], dtype=float)
    if len(q_arr) == 1:
        return np.ones(1, dtype=float)

    weights = np.empty(len(q_arr), dtype=float)
    weights[0] = 0.5 * abs(q_arr[1] - q_arr[0])
    weights[-1] = 0.5 * abs(q_arr[-1] - q_arr[-2])
    weights[1:-1] = 0.5 * np.abs(q_arr[2:] - q_arr[:-2])
    return weights


def second_derivative_periodic(values: np.ndarray, dx: float) -> np.ndarray:
    """Second derivative using a periodic centered finite difference."""
    return (np.roll(values, -1) - 2.0 * values + np.roll(values, 1)) / dx**2


def quantum_potential_from_psi(
    psi: np.ndarray,
    dx: float,
    density_floor: float = 1e-12,
) -> np.ndarray:
    """Return the Madelung/Bohm quantum potential Q = -½ ∂xx√ρ / √ρ."""
    sqrt_rho = np.sqrt(np.maximum(np.abs(psi) ** 2, density_floor))
    return -0.5 * second_derivative_periodic(sqrt_rho, dx) / sqrt_rho


def apply_quantum_potential_phase_correction(
    psi: np.ndarray,
    dx: float,
    time: float,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply a diagnostic Madelung quantum-potential phase correction."""
    Q = quantum_potential_from_psi(psi, dx)
    return psi * np.exp(-1j * strength * time * Q)


def count_caustics(snap: dict) -> int:
    """Count sign changes of the trajectory Jacobian J in a snapshot."""
    sign_J = np.sign(snap["J"])
    change = (
        (sign_J[:-1] != 0)
        & (sign_J[1:] != 0)
        & (sign_J[:-1] != sign_J[1:])
    )
    return int(np.count_nonzero(change))


def schrodinger_residual_norm(
    psi_now: np.ndarray,
    psi_prev: Optional[np.ndarray],
    dt_record: float,
    x_grid: np.ndarray,
    lam: float,
    dx: float,
) -> float:
    """Return ||i∂tψ + ½∂xxψ - Vψ||₂ for a reconstructed wavefunction."""
    if psi_prev is None or dt_record <= 0.0:
        return float("nan")
    dpsi_dt = (psi_now - psi_prev) / dt_record
    residual = (
        1.0j * dpsi_dt
        + 0.5 * second_derivative_periodic(psi_now, dx)
        - potential(x_grid, lam) * psi_now
    )
    return float(np.sqrt(np.sum(np.abs(residual) ** 2) * dx))


def fitted_correction_metrics(
    psi_ref: np.ndarray,
    psi_test: np.ndarray,
    dx: float,
) -> dict:
    """Diagnostic amplitude-only, phase-only, and oracle upper-bound metrics."""
    amp_fit = normalize(np.abs(psi_ref) * np.exp(1j * np.angle(psi_test)), dx)
    phase_fit = normalize(np.abs(psi_test) * np.exp(1j * np.angle(psi_ref)), dx)
    # The amplitude/phase identity intentionally rebuilds psi_ref from its own
    # amplitude and phase.  It is a diagnostic reference for the best possible
    # local amplitude+phase correction, so its L2/fidelity are 0/1 by design.
    amplitude_phase_identity = normalize(
        np.abs(psi_ref) * np.exp(1j * np.angle(psi_ref)), dx
    )
    return {
        "amplitude_fit_L2": l2_error(psi_ref, amp_fit, dx),
        "amplitude_fit_fidelity": fidelity(psi_ref, amp_fit, dx),
        "phase_fit_L2": l2_error(psi_ref, phase_fit, dx),
        "phase_fit_fidelity": fidelity(psi_ref, phase_fit, dx),
        "amplitude_phase_identity_L2": l2_error(
            psi_ref, amplitude_phase_identity, dx
        ),
        "amplitude_phase_identity_fidelity": fidelity(
            psi_ref, amplitude_phase_identity, dx
        ),
    }


def reconstruct_by_method(
    method: str,
    x_grid: np.ndarray,
    q_arr: np.ndarray,
    rho0_arr: np.ndarray,
    snap: dict,
    dx: float,
    time: float,
    caustic_width: float,
    smooth_sigma: float,
    packet_width: float,
    quantum_potential_strength: float,
) -> np.ndarray:
    """Dispatch one named reconstruction method."""
    if method == "raw_hj":
        return reconstruct_psi_hj(x_grid, q_arr, rho0_arr, snap)
    if method == "smoothed_hj":
        return reconstruct_psi_hj(
            x_grid,
            q_arr,
            rho0_arr,
            snap,
            caustic_width=caustic_width,
            post_smooth_sigma=smooth_sigma,
        )
    if method == "gaussian_ivr":
        return reconstruct_gaussian_ivr(
            x_grid,
            q_arr,
            rho0_arr,
            snap,
            packet_width=packet_width,
            use_hk_prefactor=False,
        )
    if method == "herman_kluk":
        return reconstruct_gaussian_ivr(
            x_grid,
            q_arr,
            rho0_arr,
            snap,
            packet_width=packet_width,
            use_hk_prefactor=True,
        )
    if method == "full_herman_kluk":
        return reconstruct_gaussian_ivr(
            x_grid,
            q_arr,
            rho0_arr,
            snap,
            packet_width=packet_width,
            use_hk_prefactor=True,
            full_hk_prefactor=True,
        )
    if method == "thawed_gaussian":
        return reconstruct_gaussian_ivr(
            x_grid,
            q_arr,
            rho0_arr,
            snap,
            packet_width=packet_width,
            thawed_width=True,
        )
    if method == "quantum_potential":
        psi = reconstruct_psi_hj(
            x_grid,
            q_arr,
            rho0_arr,
            snap,
            caustic_width=caustic_width,
            post_smooth_sigma=smooth_sigma,
        )
        return apply_quantum_potential_phase_correction(
            psi, dx, time, strength=quantum_potential_strength
        )
    raise ValueError(f"Unknown reconstruction method: {method}")


# ──────────────────────────────────────────────────────────────────────────────
# Comparison metrics
# ──────────────────────────────────────────────────────────────────────────────


def normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    """Return ψ / ||ψ||₂ (L₂ norm on the grid with spacing dx)."""
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm < 1e-30:
        return psi
    return psi / norm


def align_phase(
    psi_ref: np.ndarray,
    psi_test: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Return e^{iθ} · psi_test that minimises ||psi_ref - e^{iθ} psi_test||₂.

    The optimal global phase is θ = arg(⟨psi_ref|psi_test⟩).

    Parameters
    ----------
    psi_ref, psi_test:
        Normalized wavefunctions, shape (N,).
    dx:
        Grid spacing (for the discrete inner product).

    Returns
    -------
    psi_aligned : psi_test with the global phase removed
    """
    overlap = np.sum(np.conj(psi_ref) * psi_test) * dx
    if abs(overlap) < 1e-30:
        return psi_test.copy()
    theta = np.angle(overlap)
    return np.exp(-1j * theta) * psi_test


def l2_error(
    psi_ref: np.ndarray,
    psi_test: np.ndarray,
    dx: float,
) -> float:
    """Phase-aligned L₂ error.

    Minimise over global phase θ::

        ε = ||ψ_ref - e^{iθ} ψ_test||₂

    Both inputs should be normalised (||ψ||₂ = 1).

    Parameters
    ----------
    psi_ref, psi_test:
        Normalized wavefunctions, shape (N,).
    dx:
        Grid spacing.

    Returns
    -------
    error : float ≥ 0.  Zero means exact agreement up to a global phase.
    """
    psi_aligned = align_phase(psi_ref, psi_test, dx)
    diff = psi_ref - psi_aligned
    return float(np.sqrt(np.sum(np.abs(diff) ** 2) * dx))


def fidelity(
    psi_ref: np.ndarray,
    psi_test: np.ndarray,
    dx: float,
) -> float:
    """Fidelity F = |⟨ψ_ref|ψ_test⟩|².

    Both inputs should be normalised (||ψ||₂ = 1).

    Parameters
    ----------
    psi_ref, psi_test:
        Normalized wavefunctions, shape (N,).
    dx:
        Grid spacing.

    Returns
    -------
    fid : float in [0, 1].  One means the states are identical up to phase.
    """
    overlap = np.sum(np.conj(psi_ref) * psi_test) * dx
    return float(abs(overlap) ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────


def _write_csv(records: List[dict], path: Path) -> None:
    """Write diagnostics list to a CSV file."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info("Diagnostics written to %s", path)


def _try_plot(
    records: List[dict],
    traj_times: np.ndarray,
    traj_snaps: List[dict],
    sch_psi_list: List[np.ndarray],
    q_arr: np.ndarray,
    rho0_arr: np.ndarray,
    x_grid: np.ndarray,
    dx: float,
    out_path: Path,
    methods: List[str],
    caustic_width: float,
    smooth_sigma: float,
    packet_width: float,
    quantum_potential_strength: float,
) -> None:
    """Generate optional diagnostic plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.info("matplotlib not available — skipping plots")
        return

    # ── Plot 1: metrics vs time ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for method in methods:
        method_records = [r for r in records if r.get("method") == method]
        times = [r["t"] for r in method_records]
        errs = [r["L2_error"] for r in method_records]
        fids = [r["fidelity"] for r in method_records]
        ax1.semilogy(times, errs, lw=1.5, label=method)
        ax2.plot(times, fids, lw=1.5, label=method)

    ax1.set_ylabel("Phase-aligned L₂ error  ε(t)")
    ax1.set_title(
        "Quartic oscillator: Schrödinger vs classical reconstructions"
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_ylabel("Fidelity  F(t)")
    ax2.set_xlabel("Time  t")
    ax2.set_ylim([-0.05, 1.05])
    ax2.axhline(1.0, ls="--", color="gray", lw=0.8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    metrics_path = out_path / "metrics.png"
    fig.savefig(metrics_path, dpi=120)
    plt.close(fig)
    logger.info("Metrics plot saved to %s", metrics_path)

    # ── Plot 2: density snapshots ────────────────────────────────────────────
    n_snaps = len(traj_times)
    if n_snaps == 0:
        return

    snap_indices = [
        0,
        n_snaps // 4,
        n_snaps // 2,
        3 * n_snaps // 4,
        n_snaps - 1,
    ]
    snap_indices = sorted(set(snap_indices))

    fig, axes = plt.subplots(
        len(snap_indices),
        1,
        figsize=(9, 2.5 * len(snap_indices)),
        sharex=True,
    )
    if len(snap_indices) == 1:
        axes = [axes]

    for ax, si in zip(axes, snap_indices):
        t_snap = traj_times[si]
        snap = traj_snaps[si]
        psi_sch = normalize(sch_psi_list[si], dx)
        method = methods[0]
        psi_hj = normalize(
            reconstruct_by_method(
                method,
                x_grid,
                q_arr,
                rho0_arr,
                snap,
                dx,
                float(t_snap),
                caustic_width,
                smooth_sigma,
                packet_width,
                quantum_potential_strength,
            ),
            dx,
        )

        ax.plot(x_grid, np.abs(psi_sch) ** 2, lw=1.5, label="Schrödinger", color="tab:blue")
        ax.plot(
            x_grid,
            np.abs(psi_hj) ** 2,
            lw=1.5,
            ls="--",
            label=f"{method} reconstruction",
            color="tab:orange",
        )
        ax.set_ylabel("|ψ|²")
        ax.set_title(f"t = {t_snap:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("x")
    plt.tight_layout()
    dens_path = out_path / "density_snapshots.png"
    fig.savefig(dens_path, dpi=120)
    plt.close(fig)
    logger.info("Density snapshot plot saved to %s", dens_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────────────────────────────────────


def run_experiment(
    N: int = 512,
    L: float = 8.0,
    dt: float = 0.002,
    T: float = 3.0,
    lam: float = 0.1,
    x0: float = 0.0,
    p0: float = 1.0,
    sigma: float = 0.5,
    N_traj: int = 2000,
    methods: Optional[List[str]] = None,
    caustic_width: float = 1e-3,
    smooth_sigma: float = 0.02,
    packet_width: Optional[float] = None,
    quantum_potential_strength: float = 1.0,
    fit_corrections: bool = False,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[dict]:
    """Run the quartic-oscillator HJ vs Schrödinger experiment.

    Parameters
    ----------
    N:
        Number of Schrödinger spatial grid points.
    L:
        Half-domain size (domain is [-L, +L]).
    dt:
        Time step.
    T:
        Final time.
    lam:
        Quartic coupling λ in V(x) = ½x² + λx⁴.
    x0:
        Initial packet centre.
    p0:
        Initial packet momentum.
    sigma:
        Initial packet width.
    N_traj:
        Number of classical trajectories.
    methods:
        Reconstruction methods to compare.  Available methods are
        ``raw_hj``, ``smoothed_hj``, ``gaussian_ivr``, ``herman_kluk``,
        ``full_herman_kluk``, ``thawed_gaussian``, and ``quantum_potential``.
    caustic_width:
        Jacobian regularization used by smoothed methods.
    smooth_sigma:
        Post-reconstruction Gaussian smoothing width in x-units.
    packet_width:
        Finite coherent-packet width for IVR methods.  Defaults to ``sigma``.
    quantum_potential_strength:
        Strength of the Madelung quantum-potential phase correction.
    fit_corrections:
        If true, add diagnostic amplitude/phase/local-fit metrics.
    out_dir:
        If given, write ``diagnostics.csv`` and optional plots there.
    verbose:
        Print progress to stderr/stdout.

    Returns
    -------
    records : list of dicts with one row per time and reconstruction method.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    x_grid = np.linspace(-L, L, N, endpoint=False)
    dx = x_grid[1] - x_grid[0]
    n_steps = max(1, int(round(T / dt)))
    # Bound memory/plot size while retaining enough temporal resolution for
    # residual diagnostics.
    record_every = max(1, n_steps // SNAPSHOT_RECORD_LIMIT)
    n_records = n_steps // record_every
    if methods is None:
        methods = ["raw_hj"]
    methods = list(dict.fromkeys(methods))
    packet_width_value = sigma if packet_width is None else packet_width

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        logger.info(
            "\n=== Quartic-oscillator Schrödinger vs HJ reconstruction ===\n"
            "  V(x) = 0.5x² + %.3fx⁴  (m = ℏ = ω = 1)\n"
            "  Grid    : N=%d, L=%.2f, dx=%.4f\n"
            "  Time    : T=%.2f, dt=%.4f, steps=%d, snapshots=%d\n"
            "  Packet  : x₀=%.2f, p₀=%.2f, σ=%.2f\n"
            "  Traj.   : N_traj=%d\n"
            "  Methods : %s\n",
            lam,
            N,
            L,
            dx,
            T,
            dt,
            n_steps,
            n_records,
            x0,
            p0,
            sigma,
            N_traj,
            ", ".join(methods),
        )

    # ── Initial wavefunction ──────────────────────────────────────────────────
    psi0 = normalize(initial_gaussian(x_grid, x0, p0, sigma), dx)

    # ── Trajectory initial conditions ─────────────────────────────────────────
    # Use a slightly wider domain to reduce boundary artifacts
    q_arr = np.linspace(-L * 1.1, L * 1.1, N_traj)
    rho0_arr = initial_density(q_arr, x0, sigma)
    S0_arr = q_arr * p0      # S₀(q) = p₀ · q
    p0_arr = initial_action_deriv(q_arr, p0)  # p(0;q) = S₀'(q) = p₀

    # ── Schrödinger evolution ─────────────────────────────────────────────────
    if verbose:
        logger.info("Running split-operator Schrödinger solver …")
    sch_times, sch_psi_list = run_schrodinger(
        psi0, x_grid, lam, dt, n_steps, record_every=record_every
    )

    # ── Classical trajectory evolution ────────────────────────────────────────
    if verbose:
        logger.info("Running classical trajectory ODE (vectorized RK4) …")
    traj_times, traj_snaps = run_trajectories(
        q_arr, p0_arr, S0_arr, lam, dt, n_steps, record_every=record_every
    )

    # ── Compare at each recorded time ─────────────────────────────────────────
    if verbose:
        logger.info("Computing HJ reconstruction and metrics …")

    records: List[dict] = []
    # t = 0: exact by construction for all method baselines.
    for method in methods:
        row = {
            "t": 0.0,
            "method": method,
            "L2_error": 0.0,
            "fidelity": 1.0,
            "norm": 1.0,
            "caustic_count": 0,
            "max_density": float(np.max(np.abs(psi0) ** 2)),
            "residual_norm": float("nan"),
        }
        if fit_corrections:
            row.update(
                {
                    "amplitude_fit_L2": 0.0,
                    "amplitude_fit_fidelity": 1.0,
                    "phase_fit_L2": 0.0,
                    "phase_fit_fidelity": 1.0,
                    "amplitude_phase_identity_L2": 0.0,
                    "amplitude_phase_identity_fidelity": 1.0,
                }
            )
        records.append(row)

    report_stride = max(1, n_records // 10)
    prev_psi_by_method: dict[str, Optional[np.ndarray]] = {
        method: None for method in methods
    }
    dt_record = record_every * dt

    for i, (t, snap, psi_sch) in enumerate(
        zip(traj_times, traj_snaps, sch_psi_list)
    ):
        psi_sch_n = normalize(psi_sch, dx)
        caustics = count_caustics(snap)
        log_parts = []

        for method in methods:
            psi_raw = reconstruct_by_method(
                method,
                x_grid,
                q_arr,
                rho0_arr,
                snap,
                dx,
                float(t),
                caustic_width,
                smooth_sigma,
                packet_width_value,
                quantum_potential_strength,
            )
            raw_norm = float(np.sqrt(np.sum(np.abs(psi_raw) ** 2) * dx))
            psi_n = normalize(psi_raw, dx)

            err = l2_error(psi_sch_n, psi_n, dx)
            fid = fidelity(psi_sch_n, psi_n, dx)
            residual = schrodinger_residual_norm(
                psi_n,
                prev_psi_by_method[method],
                dt_record,
                x_grid,
                lam,
                dx,
            )
            prev_psi_by_method[method] = psi_n.copy()

            row = {
                "t": float(t),
                "method": method,
                "L2_error": err,
                "fidelity": fid,
                "norm": raw_norm,
                "caustic_count": caustics,
                "max_density": float(np.max(np.abs(psi_n) ** 2)),
                "residual_norm": residual,
            }
            if fit_corrections:
                row.update(fitted_correction_metrics(psi_sch_n, psi_n, dx))
            records.append(row)
            log_parts.append(f"{method}: L2={err:.3e}, F={fid:.5f}")

        if verbose and (i % report_stride == 0 or i == n_records - 1):
            logger.info("  t=%6.3f  %s", t, " | ".join(log_parts))

    # ── Output ────────────────────────────────────────────────────────────────
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        _write_csv(records, out_path / "diagnostics.csv")
        _try_plot(
            records,
            traj_times,
            traj_snaps,
            sch_psi_list,
            q_arr,
            rho0_arr,
            x_grid,
            dx,
            out_path,
            methods,
            caustic_width,
            smooth_sigma,
            packet_width_value,
            quantum_potential_strength,
        )

    return records


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Quartic-oscillator Schrödinger vs Hamilton–Jacobi stress test.\n"
            "Compares split-operator Schrödinger evolution against a classical\n"
            "multibranch action-density reconstruction and reports phase-aligned\n"
            "L₂ error and fidelity over time."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--N",
        type=int,
        default=512,
        metavar="INT",
        help="Schrödinger grid points (default: %(default)s)",
    )
    p.add_argument(
        "--L",
        type=float,
        default=8.0,
        metavar="FLOAT",
        help="Half-domain size: grid is [-L, L] (default: %(default)s)",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.002,
        metavar="FLOAT",
        help="Time step (default: %(default)s)",
    )
    p.add_argument(
        "--T",
        type=float,
        default=3.0,
        metavar="FLOAT",
        help="Final time (default: %(default)s)",
    )
    p.add_argument(
        "--lambda_",
        "--lambda",
        dest="lam",
        type=float,
        default=0.1,
        metavar="FLOAT",
        help="Quartic coupling λ in V = ½x² + λx⁴ (default: %(default)s)",
    )
    p.add_argument(
        "--x0",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Initial packet center (default: %(default)s)",
    )
    p.add_argument(
        "--p0",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Initial packet momentum (default: %(default)s)",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Initial packet width σ (default: %(default)s)",
    )
    p.add_argument(
        "--N-traj",
        dest="N_traj",
        type=int,
        default=2000,
        metavar="INT",
        help="Number of classical trajectories (default: %(default)s)",
    )
    p.add_argument(
        "--method",
        dest="methods",
        action="append",
        choices=[
            "raw_hj",
            "smoothed_hj",
            "gaussian_ivr",
            "herman_kluk",
            "full_herman_kluk",
            "thawed_gaussian",
            "quantum_potential",
        ],
        default=None,
        help=(
            "Reconstruction method to run. May be repeated. "
            "Default: raw_hj"
        ),
    )
    p.add_argument(
        "--compare-all",
        action="store_true",
        help="Run all reconstruction methods on the same trajectory data",
    )
    p.add_argument(
        "--caustic-width",
        type=float,
        default=1e-3,
        metavar="FLOAT",
        help="Jacobian regularization width for caustic smoothing (default: %(default)s)",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.02,
        metavar="FLOAT",
        help="Post-reconstruction Gaussian smoothing width in x-units (default: %(default)s)",
    )
    p.add_argument(
        "--packet-width",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Gaussian IVR packet width; defaults to initial sigma",
    )
    p.add_argument(
        "--quantum-potential-strength",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Strength for quantum-potential phase correction (default: %(default)s)",
    )
    p.add_argument(
        "--fit-corrections",
        action="store_true",
        help="Add amplitude-only, phase-only, and local-fit diagnostic metrics",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output directory for diagnostics.csv and plots (optional)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    methods = args.methods
    if args.compare_all:
        methods = [
            "raw_hj",
            "smoothed_hj",
            "gaussian_ivr",
            "herman_kluk",
            "full_herman_kluk",
            "thawed_gaussian",
            "quantum_potential",
        ]
    records = run_experiment(
        N=args.N,
        L=args.L,
        dt=args.dt,
        T=args.T,
        lam=args.lam,
        x0=args.x0,
        p0=args.p0,
        sigma=args.sigma,
        N_traj=args.N_traj,
        methods=methods,
        caustic_width=args.caustic_width,
        smooth_sigma=args.smooth_sigma,
        packet_width=args.packet_width,
        quantum_potential_strength=args.quantum_potential_strength,
        fit_corrections=args.fit_corrections,
        out_dir=args.out_dir,
        verbose=not args.quiet,
    )

    # Print final summary
    if records:
        final_t = records[-1]["t"]
        print(f"\nFinal metrics at t={final_t:.4f}:")
        for row in records:
            if row["t"] == final_t:
                print(
                    f"  {row['method']}: "
                    f"L2_error={row['L2_error']:.4e}  "
                    f"fidelity={row['fidelity']:.6f}  "
                    f"norm={row['norm']:.6f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
