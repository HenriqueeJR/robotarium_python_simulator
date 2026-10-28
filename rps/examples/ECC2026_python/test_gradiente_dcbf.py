from __future__ import annotations

import numpy as np

from .build_local_dcbf import build_local_dcbf


def test_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    x_vec = np.arange(-1.0, 1.001, 0.04)
    y_vec = np.arange(-1.0, 1.001, 0.04)
    X, Y = np.meshgrid(x_vec, y_vec)
    field = {
        "x_vec": x_vec,
        "y_vec": y_vec,
        "ux": 0.075 + 0.012 * X - 0.006 * Y,
        "uy": 0.018 + 0.004 * X + 0.010 * Y,
    }
    horizon = 9
    params = {
        "N": horizon, "Ts": 0.08,
        "x0": np.array([-0.18, 0.09, 0.12]), "field_gain": 1.0,
        "W_track": 40.0, "W_goal": 0.8, "W_terminal_goal": 15.0,
        "target": np.array([0.45, 0.02]), "robot_radius": 0.05,
        "rho_geofence": 8.0e3, "R_v": 0.05, "R_w": 0.03,
        "R_dv": 0.08, "R_dw": 0.05,
        "u_previous": np.array([0.055, -0.02]),
        "d_safe": 0.22, "gamma_cbf": 0.10, "rho_cbf": 2.0e3,
        "N_blocks": 1,
        "blocks_params": np.array([-0.15, 0.15, -0.05, 0.05]),
        "rotatorias": np.empty((0, 4)), "h_p2s": 0.05,
        "eta_safe": 8.0e3, "gamma_safe": 0.5,
        "eta_obs": 8.0e3, "gamma_obs": 0.5,
        "neighbor_positions": np.zeros((2, horizon + 1, 1)),
    }
    params["neighbor_positions"][0, :, 0] = np.linspace(0.105, 0.045, horizon + 1)
    params["neighbor_positions"][1, :, 0] = np.linspace(0.015, 0.010, horizon + 1)
    controls = np.vstack((
        0.075 + 0.008 * rng.standard_normal(horizon),
        0.04 * rng.standard_normal(horizon),
    ))
    U = controls.reshape(-1, order="F")
    cost, gradient, _, diagnostics = build_local_dcbf(U, params, field)

    epsilon = 1e-6
    finite_difference = np.empty_like(U)
    for index in range(U.size):
        direction = np.zeros_like(U)
        direction[index] = epsilon
        cost_plus = build_local_dcbf(U + direction, params, field, False)
        cost_minus = build_local_dcbf(U - direction, params, field, False)
        finite_difference[index] = (cost_plus - cost_minus) / (2.0 * epsilon)
    relative_error = np.linalg.norm(gradient - finite_difference) / max(
        1.0, np.linalg.norm(finite_difference)
    )
    assert diagnostics["max_dcbf_violation"] > 1e-8
    assert diagnostics["max_geofence_violation"] > 1e-8
    assert relative_error < 5e-5, (cost, relative_error)
