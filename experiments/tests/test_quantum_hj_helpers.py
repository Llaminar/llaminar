"""
Tests for numerical helpers in experiments/quantum_hj_quartic.py
=================================================================

Run with::

    pytest experiments/tests/test_quantum_hj_helpers.py -v

These tests validate the correctness of isolated numerical helpers used by the
quartic-oscillator stress test (not the full end-to-end experiment, which is
slow).  They cover:

- Potential and its derivatives (checked against numerical differentiation)
- Normalization
- Phase alignment
- Fidelity metric (identity and orthogonality cases)
- Schrödinger norm conservation for short evolution
- t=0 HJ reconstruction accuracy (must reproduce the initial wavefunction)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the experiments package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.quantum_hj_quartic import (  # noqa: E402
    align_phase,
    count_caustics,
    fidelity,
    fitted_correction_metrics,
    gaussian_smooth_periodic,
    initial_density,
    initial_gaussian,
    l2_error,
    normalize,
    potential,
    potential_first_deriv,
    potential_second_deriv,
    quantum_potential_from_psi,
    reconstruct_by_method,
    reconstruct_gaussian_ivr,
    reconstruct_psi_hj,
    run_schrodinger,
    second_derivative_periodic,
    trajectory_quadrature_weights,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_RNG = np.random.default_rng(42)

LAM_DEFAULT = 0.1


def _grid(N: int = 128, L: float = 6.0) -> tuple[np.ndarray, float]:
    x = np.linspace(-L, L, N, endpoint=False)
    return x, x[1] - x[0]


# ──────────────────────────────────────────────────────────────────────────────
# Potential and derivatives
# ──────────────────────────────────────────────────────────────────────────────


class TestPotential:
    """Validate V, V', V'' analytically and against finite differences."""

    def test_potential_at_origin(self):
        assert potential(np.array([0.0]), LAM_DEFAULT)[0] == pytest.approx(0.0)

    def test_potential_even_symmetry(self):
        xs = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            potential(xs, LAM_DEFAULT), potential(-xs, LAM_DEFAULT)
        )

    def test_first_deriv_finite_diff(self):
        xs = np.linspace(-3.0, 3.0, 40)
        h = 1e-5
        Vp_analytic = potential_first_deriv(xs, LAM_DEFAULT)
        Vp_numeric = (potential(xs + h, LAM_DEFAULT) - potential(xs - h, LAM_DEFAULT)) / (2 * h)
        np.testing.assert_allclose(Vp_analytic, Vp_numeric, rtol=1e-5)

    def test_second_deriv_finite_diff(self):
        xs = np.linspace(-3.0, 3.0, 40)
        h = 1e-4
        Vpp_analytic = potential_second_deriv(xs, LAM_DEFAULT)
        Vpp_numeric = (
            potential(xs + h, LAM_DEFAULT)
            - 2 * potential(xs, LAM_DEFAULT)
            + potential(xs - h, LAM_DEFAULT)
        ) / h**2
        np.testing.assert_allclose(Vpp_analytic, Vpp_numeric, rtol=1e-4)

    def test_second_deriv_positive_definite(self):
        """V''(x) = 1 + 12λx² ≥ 1 > 0 for λ ≥ 0 (ensures a confining potential)."""
        xs = np.linspace(-5.0, 5.0, 100)
        assert np.all(potential_second_deriv(xs, LAM_DEFAULT) > 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalize:
    def test_unit_norm_after_normalize(self):
        x, dx = _grid()
        psi = initial_gaussian(x, 0.0, 1.0, 0.5)
        psi_n = normalize(psi, dx)
        norm = np.sqrt(np.sum(np.abs(psi_n) ** 2) * dx)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_phase_preserved(self):
        x, dx = _grid()
        psi = initial_gaussian(x, 0.0, 1.0, 0.5)
        psi_n = normalize(psi, dx)
        # Dividing by a real positive scalar should not change the phase pattern
        ratio = psi_n / psi
        # All ratios should have the same argument (constant real positive scalar)
        np.testing.assert_allclose(np.imag(ratio), 0.0, atol=1e-12)

    def test_zero_input_returns_input(self):
        x, dx = _grid()
        psi = np.zeros(len(x), dtype=complex)
        out = normalize(psi, dx)
        np.testing.assert_array_equal(out, psi)


# ──────────────────────────────────────────────────────────────────────────────
# Phase alignment
# ──────────────────────────────────────────────────────────────────────────────


class TestAlignPhase:
    def test_identical_returns_same(self):
        x, dx = _grid()
        psi = normalize(initial_gaussian(x, 0.0, 1.0, 0.5), dx)
        aligned = align_phase(psi, psi, dx)
        np.testing.assert_allclose(np.abs(aligned - psi), 0.0, atol=1e-14)

    def test_known_phase_removed(self):
        x, dx = _grid()
        psi = normalize(initial_gaussian(x, 0.0, 1.0, 0.5), dx)
        theta = 1.2345
        psi_rotated = np.exp(1j * theta) * psi
        aligned = align_phase(psi, psi_rotated, dx)
        np.testing.assert_allclose(np.abs(aligned - psi), 0.0, atol=1e-12)

    def test_l2_error_zero_for_phase_offset(self):
        x, dx = _grid()
        psi = normalize(initial_gaussian(x, 0.5, 0.5, 0.6), dx)
        for theta in [0.0, 0.3, np.pi, -2.7]:
            err = l2_error(psi, np.exp(1j * theta) * psi, dx)
            assert err == pytest.approx(0.0, abs=1e-12)

    def test_l2_error_nonzero_for_different_states(self):
        x, dx = _grid()
        psi1 = normalize(initial_gaussian(x, 0.0, 0.0, 0.5), dx)
        psi2 = normalize(initial_gaussian(x, 2.0, 0.0, 0.5), dx)
        err = l2_error(psi1, psi2, dx)
        assert err > 0.1  # significantly different states


# ──────────────────────────────────────────────────────────────────────────────
# Fidelity
# ──────────────────────────────────────────────────────────────────────────────


class TestFidelity:
    def test_identical_fidelity_one(self):
        x, dx = _grid()
        psi = normalize(initial_gaussian(x, 0.0, 1.0, 0.5), dx)
        assert fidelity(psi, psi, dx) == pytest.approx(1.0, abs=1e-12)

    def test_phase_shifted_fidelity_one(self):
        x, dx = _grid()
        psi = normalize(initial_gaussian(x, 0.0, 1.0, 0.5), dx)
        psi_rotated = np.exp(1j * 2.718) * psi
        assert fidelity(psi, psi_rotated, dx) == pytest.approx(1.0, abs=1e-12)

    def test_orthogonal_fidelity_near_zero(self):
        """Two well-separated Gaussians should have near-zero fidelity."""
        x, dx = _grid(N=256, L=10.0)
        psi1 = normalize(initial_gaussian(x, -5.0, 0.0, 0.3), dx)
        psi2 = normalize(initial_gaussian(x, +5.0, 0.0, 0.3), dx)
        assert fidelity(psi1, psi2, dx) < 1e-10

    def test_fidelity_bounded(self):
        x, dx = _grid()
        psi1 = normalize(initial_gaussian(x, 0.0, 0.0, 0.5), dx)
        psi2 = normalize(initial_gaussian(x, 0.5, 1.0, 0.7), dx)
        fid = fidelity(psi1, psi2, dx)
        assert 0.0 <= fid <= 1.0 + 1e-12


# ──────────────────────────────────────────────────────────────────────────────
# Schrödinger solver: norm conservation
# ──────────────────────────────────────────────────────────────────────────────


class TestSchrodingerNormConservation:
    """The split-operator method should conserve ||ψ||₂ to near machine precision."""

    @pytest.mark.parametrize("lam", [0.0, 0.1, 0.5])
    def test_norm_conserved(self, lam):
        x, dx = _grid(N=256, L=8.0)
        psi0 = normalize(initial_gaussian(x, 0.0, 1.0, 0.5), dx)
        # 100 steps of dt=0.01
        _times, psi_list = run_schrodinger(
            psi0, x, lam=lam, dt=0.01, n_steps=100, record_every=50
        )
        for psi in psi_list:
            norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
            assert norm == pytest.approx(1.0, abs=1e-6), (
                f"Norm not conserved for λ={lam}: got {norm:.8f}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# HJ reconstruction at t = 0
# ──────────────────────────────────────────────────────────────────────────────


class TestHJt0Reconstruction:
    """At t=0 the HJ reconstruction must reproduce the initial Gaussian
    (up to grid interpolation error), since x(0;q)=q and J(0)=1."""

    def test_t0_reconstruction_matches_initial(self):
        x, dx = _grid(N=256, L=6.0)
        x0, p0, sigma = 0.0, 0.5, 0.5
        q_arr = np.linspace(-6.0 * 1.1, 6.0 * 1.1, 2000)
        rho0_arr = initial_density(q_arr, x0, sigma)

        # Build a t=0 snapshot: trivial identity map
        snap = {
            "x": q_arr.copy(),
            "S": q_arr * p0,  # S₀(q) = p₀·q
            "J": np.ones_like(q_arr),
            "P": np.zeros_like(q_arr),
            "mu": np.zeros(len(q_arr), dtype=int),
        }

        psi_hj = reconstruct_psi_hj(x, q_arr, rho0_arr, snap)
        psi_ref = initial_gaussian(x, x0, p0, sigma)

        psi_hj_n = normalize(psi_hj, dx)
        psi_ref_n = normalize(psi_ref, dx)

        err = l2_error(psi_ref_n, psi_hj_n, dx)
        # Interpolation error should be small (~O(dq²)); we allow 5%
        assert err < 0.05, (
            f"t=0 HJ reconstruction L2 error too large: {err:.4f}.  "
            "Check reconstruct_psi_hj or initial conditions."
        )

    def test_t0_fidelity_near_one(self):
        x, dx = _grid(N=256, L=6.0)
        x0, p0, sigma = 0.5, 0.3, 0.6
        q_arr = np.linspace(-6.6, 6.6, 3000)
        rho0_arr = initial_density(q_arr, x0, sigma)

        snap = {
            "x": q_arr.copy(),
            "S": q_arr * p0,
            "J": np.ones_like(q_arr),
            "P": np.zeros_like(q_arr),
            "mu": np.zeros(len(q_arr), dtype=int),
        }

        psi_hj = normalize(reconstruct_psi_hj(x, q_arr, rho0_arr, snap), dx)
        psi_ref = normalize(initial_gaussian(x, x0, p0, sigma), dx)

        fid = fidelity(psi_ref, psi_hj, dx)
        assert fid > 0.99, (
            f"t=0 HJ fidelity too low: {fid:.6f}.  "
            "Check reconstruct_psi_hj or initial conditions."
        )


class TestReconstructionVariants:
    def _t0_snapshot(self, q_arr, p0):
        return {
            "x": q_arr.copy(),
            "p": np.full_like(q_arr, p0),
            "S": q_arr * p0,
            "J": np.ones_like(q_arr),
            "P": np.zeros_like(q_arr),
            "mu": np.zeros(len(q_arr), dtype=int),
        }

    def test_caustic_regularization_bounds_amplitude(self):
        x, _ = _grid(N=128, L=4.0)
        q_arr = np.linspace(-4.0, 4.0, 400)
        rho0_arr = initial_density(q_arr, 0.0, 0.5)
        snap = self._t0_snapshot(q_arr, p0=0.0)
        snap["J"][len(q_arr) // 2] = 0.0

        psi_raw = reconstruct_psi_hj(x, q_arr, rho0_arr, snap)
        psi_smooth = reconstruct_psi_hj(
            x, q_arr, rho0_arr, snap, caustic_width=1e-2
        )

        assert np.max(np.abs(psi_smooth)) < np.max(np.abs(psi_raw))

    def test_gaussian_smoothing_reduces_high_frequency_component(self):
        x, dx = _grid(N=128, L=4.0)
        values = np.exp(1j * 20.0 * x)
        smoothed = gaussian_smooth_periodic(values, dx, sigma=0.1)
        assert np.max(np.abs(smoothed)) < np.max(np.abs(values))

    def test_gaussian_ivr_returns_finite_nonzero_wavefunction(self):
        x, dx = _grid(N=128, L=4.0)
        q_arr = np.linspace(-4.0, 4.0, 600)
        rho0_arr = initial_density(q_arr, 0.0, 0.5)
        snap = self._t0_snapshot(q_arr, p0=0.3)

        psi = reconstruct_gaussian_ivr(x, q_arr, rho0_arr, snap, packet_width=0.5)
        psi_n = normalize(psi, dx)

        assert np.all(np.isfinite(psi_n))
        assert np.sqrt(np.sum(np.abs(psi_n) ** 2) * dx) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize(
        "method",
        ["raw_hj", "smoothed_hj", "gaussian_ivr", "herman_kluk", "quantum_potential"],
    )
    def test_reconstruct_by_method_dispatches(self, method):
        x, dx = _grid(N=96, L=4.0)
        q_arr = np.linspace(-4.0, 4.0, 300)
        rho0_arr = initial_density(q_arr, 0.0, 0.5)
        snap = self._t0_snapshot(q_arr, p0=0.2)

        psi = reconstruct_by_method(
            method,
            x,
            q_arr,
            rho0_arr,
            snap,
            dx,
            time=0.0,
            caustic_width=1e-3,
            smooth_sigma=0.02,
            packet_width=0.5,
            quantum_potential_strength=1.0,
        )

        assert np.all(np.isfinite(psi))
        assert np.max(np.abs(psi)) > 0.0

    def test_count_caustics_detects_jacobian_sign_changes(self):
        snap = {"J": np.array([1.0, 0.5, -0.2, -0.1, 0.4])}
        assert count_caustics(snap) == 2


class TestDiagnostics:
    def test_second_derivative_periodic_matches_sine(self):
        x, dx = _grid(N=256, L=np.pi)
        values = np.sin(2.0 * x)
        d2 = second_derivative_periodic(values, dx)
        np.testing.assert_allclose(d2, -4.0 * values, atol=5e-3)

    def test_quantum_potential_finite_for_gaussian(self):
        x, dx = _grid(N=256, L=6.0)
        psi = normalize(initial_gaussian(x, 0.0, 0.0, 0.5), dx)
        Q = quantum_potential_from_psi(psi, dx)
        assert np.all(np.isfinite(Q))

    def test_fitted_correction_metrics_are_nontrivial_for_distinct_states(self):
        x, dx = _grid(N=128, L=5.0)
        psi_ref = normalize(initial_gaussian(x, 0.0, 0.0, 0.5), dx)
        psi_test = normalize(initial_gaussian(x, 0.5, 1.0, 0.7), dx)

        metrics = fitted_correction_metrics(psi_ref, psi_test, dx)

        for key, value in metrics.items():
            assert np.isfinite(value), key
        assert metrics["amplitude_fit_L2"] > 0.0
        assert metrics["phase_fit_L2"] > 0.0
        assert metrics["amplitude_phase_identity_L2"] == pytest.approx(
            0.0, abs=1e-12
        )
        assert metrics["amplitude_phase_identity_fidelity"] == pytest.approx(
            1.0, abs=1e-12
        )

    def test_trajectory_quadrature_weights_support_nonuniform_grid(self):
        """Trapezoidal weights: half-width edges, centered spans internally."""
        q = np.array([0.0, 0.5, 2.0, 3.0])
        weights = trajectory_quadrature_weights(q)
        np.testing.assert_allclose(weights, [0.25, 1.0, 1.25, 0.5])
