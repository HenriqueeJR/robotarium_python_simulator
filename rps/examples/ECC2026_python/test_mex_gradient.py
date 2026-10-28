from __future__ import annotations

import numpy as np

from .diagnostico_comparacao_mex import diagnostico_comparacao_mex


def test_gradient_matches_finite_differences():
    horizon = 3
    x_vec = np.linspace(-1.0, 1.0, 11)
    y_vec = np.linspace(-1.0, 1.0, 11)
    X, Y = np.meshgrid(x_vec, y_vec)
    field = {
        "x_vec": x_vec, "y_vec": y_vec,
        "ux": 0.05 + 0.01 * X, "uy": 0.02 + 0.01 * Y,
    }
    params = {
        "N": horizon, "Ts": 0.05, "x0": np.array([-0.5, 0.0, 0.0]),
        "neighbor_positions": np.empty((2, horizon + 1, 0)),
        "u_previous": np.array([0.03, 0.0]), "target": np.array([0.5, 0.0]),
        "d_safe": 0.2, "gamma_cbf": 0.1, "rho_cbf": 100.0,
        "rotatorias": np.empty((0, 4)), "N_blocks": 0,
        "blocks_params": np.empty(0), "h_p2s": 0.05,
        "eta_safe": 0.0, "gamma_safe": 0.5, "eta_obs": 0.0,
        "gamma_obs": 0.5, "field_gain": 1.0, "W_track": 10.0,
        "W_goal": 1.0, "W_terminal_goal": 5.0, "robot_radius": 0.05,
        "rho_geofence": 0.0, "R_v": 0.01, "R_w": 0.01,
        "R_dv": 0.05, "R_dw": 0.1,
    }
    U = np.array([0.04, 0.01, 0.05, -0.02, 0.03, 0.02])
    report = diagnostico_comparacao_mex(U, params, field)
    assert report["passed"], report["gradient_relative_error"]
