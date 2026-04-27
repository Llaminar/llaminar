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

3. Evolve the Jacobian variational equations::

       dJ/dt = P,   dP/dt = -V''(x) J
       where J = ∂x(t;q)/∂q,  P = ∂p(t;q)/∂q
       Initial: J(0)=1, P(0)=∂²S₀/∂q² (= 0 for uniform initial momentum)

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

        state = [x₁..xₙ | p₁..pₙ | S₁..Sₙ | J₁..Jₙ | P₁..Pₙ]

    Equations (m = ℏ = 1)::

        dx/dt = p
        dp/dt = -V'(x)
        dS/dt = p²/2 - V(x)   (Lagrangian)
        dJ/dt = P
        dP/dt = -V''(x) · J

    Parameters
    ----------
    state:
        Concatenated state array of length 5·N_traj.
    lam:
        Quartic coupling λ.

    Returns
    -------
    rhs : array of length 5·N_traj
    """
    n = len(state) // 5
    x = state[0:n]
    p = state[n : 2 * n]
    J = state[3 * n : 4 * n]
    P = state[4 * n : 5 * n]

    Vp = potential_first_deriv(x, lam)
    Vpp = potential_second_deriv(x, lam)
    V = potential(x, lam)

    return np.concatenate(
        [
            p,               # dx/dt
            -Vp,             # dp/dt
            0.5 * p**2 - V,  # dS/dt  (Lagrangian)
            P,               # dJ/dt
            -Vpp * J,        # dP/dt
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

    Also integrates the Jacobian variational equations (J, P) and tracks
    the Maslov index for each trajectory.

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
        ``x``, ``p``, ``S``, ``J``, ``P`` (shape (N_traj,)) and
        ``mu`` (integer array, shape (N_traj,))
    """
    n = len(q_arr)
    # J(0) = 1 (identity Lagrangian map), P(0) = ∂p₀/∂q = 0 (uniform initial momentum)
    J0 = np.ones(n)
    P0 = np.zeros(n)

    state = np.concatenate([q_arr, p0_arr, S0_arr, J0, P0])

    # Maslov index: increment by 1 at each caustic (J sign change)
    mu = np.zeros(n, dtype=int)
    prev_sign_J = np.sign(J0)  # initially all +1

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
                    "J": state[3 * n : 4 * n].copy(),
                    "P": state[4 * n : 5 * n].copy(),
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

        # Avoid division by zero near caustics
        safe_J = np.maximum(absJ_at_x, 1e-30)
        amplitude = np.sqrt(np.maximum(rho0_at_x / safe_J, 0.0))
        phase = S_at_x - 0.5 * np.pi * mu_at_x

        psi[mask] += amplitude * np.exp(1j * phase)

    return psi


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
) -> None:
    """Generate optional diagnostic plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.info("matplotlib not available — skipping plots")
        return

    # ── Plot 1: metrics vs time ──────────────────────────────────────────────
    times = [r["t"] for r in records]
    errs = [r["L2_error"] for r in records]
    fids = [r["fidelity"] for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.semilogy(times, errs, color="tab:red", lw=1.5)
    ax1.set_ylabel("Phase-aligned L₂ error  ε(t)")
    ax1.set_title(
        "Quartic oscillator: Schrödinger vs classical HJ reconstruction"
    )
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(times, fids, color="tab:blue", lw=1.5)
    ax2.set_ylabel("Fidelity  F(t) = |⟨ψ_Sch|ψ_HJ⟩|²")
    ax2.set_xlabel("Time  t")
    ax2.set_ylim([-0.05, 1.05])
    ax2.axhline(1.0, ls="--", color="gray", lw=0.8)
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
        psi_hj = normalize(reconstruct_psi_hj(x_grid, q_arr, rho0_arr, snap), dx)

        ax.plot(x_grid, np.abs(psi_sch) ** 2, lw=1.5, label="Schrödinger", color="tab:blue")
        ax.plot(
            x_grid,
            np.abs(psi_hj) ** 2,
            lw=1.5,
            ls="--",
            label="HJ reconstruction",
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
    out_dir:
        If given, write ``diagnostics.csv`` and optional plots there.
    verbose:
        Print progress to stderr/stdout.

    Returns
    -------
    records : list of dicts, each with keys ``t``, ``L2_error``, ``fidelity``.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    x_grid = np.linspace(-L, L, N, endpoint=False)
    dx = x_grid[1] - x_grid[0]
    n_steps = max(1, int(round(T / dt)))
    # Aim for ~200 recorded snapshots (never more than n_steps)
    record_every = max(1, n_steps // 200)
    n_records = n_steps // record_every

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        logger.info(
            "\n=== Quartic-oscillator Schrödinger vs HJ reconstruction ===\n"
            "  V(x) = 0.5x² + %.3fx⁴  (m = ℏ = ω = 1)\n"
            "  Grid    : N=%d, L=%.2f, dx=%.4f\n"
            "  Time    : T=%.2f, dt=%.4f, steps=%d, snapshots=%d\n"
            "  Packet  : x₀=%.2f, p₀=%.2f, σ=%.2f\n"
            "  Traj.   : N_traj=%d\n",
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
    # t = 0: exact by construction (reconstruction = initial wavefunction)
    records.append({"t": 0.0, "L2_error": 0.0, "fidelity": 1.0})

    report_stride = max(1, n_records // 10)

    for i, (t, snap, psi_sch) in enumerate(
        zip(traj_times, traj_snaps, sch_psi_list)
    ):
        psi_hj_raw = reconstruct_psi_hj(x_grid, q_arr, rho0_arr, snap)
        psi_sch_n = normalize(psi_sch, dx)
        psi_hj_n = normalize(psi_hj_raw, dx)

        err = l2_error(psi_sch_n, psi_hj_n, dx)
        fid = fidelity(psi_sch_n, psi_hj_n, dx)

        records.append({"t": float(t), "L2_error": err, "fidelity": fid})

        if verbose and (i % report_stride == 0 or i == n_records - 1):
            logger.info("  t=%6.3f  L2_error=%.4e  fidelity=%.6f", t, err, fid)

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
        out_dir=args.out_dir,
        verbose=not args.quiet,
    )

    # Print final summary
    if records:
        final = records[-1]
        print(
            f"\nFinal metrics at t={final['t']:.4f}: "
            f"L2_error={final['L2_error']:.4e}  "
            f"fidelity={final['fidelity']:.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
