from __future__ import annotations

import numpy as np


def make_local_params(
    x0, neighbor_positions, u_previous, target, N, Ts, d_safe, gamma_cbf,
    rho_cbf, robot_radius, corredores, rotatorias,
):
    distance_to_goal = np.linalg.norm(np.asarray(x0)[:2] - np.asarray(target))
    near_goal_weight = np.exp(-(distance_to_goal / 0.25) ** 2)
    blocks = []
    for corridor in corredores:
        vertices = np.asarray(corridor["v"])
        blocks.extend((
            np.min(vertices[:, 0]) + robot_radius,
            np.max(vertices[:, 0]) - robot_radius,
            np.min(vertices[:, 1]) + robot_radius,
            np.max(vertices[:, 1]) - robot_radius,
        ))
    return {
        "N": int(N), "Ts": float(Ts),
        "x0": np.asarray(x0, dtype=float).reshape(3),
        "neighbor_positions": np.asarray(neighbor_positions, dtype=float),
        "u_previous": np.asarray(u_previous, dtype=float).reshape(2),
        "target": np.asarray(target, dtype=float).reshape(2),
        "d_safe": float(d_safe), "gamma_cbf": float(gamma_cbf),
        "rho_cbf": float(rho_cbf),
        "rotatorias": np.asarray(rotatorias, dtype=float).reshape(-1, 5)[:, :4],
        "N_blocks": len(corredores),
        "blocks_params": np.asarray(blocks, dtype=float),
        "h_p2s": 0.05, "eta_safe": 1.0e7, "gamma_safe": 0.5,
        "eta_obs": 1.0e7, "gamma_obs": 0.5,
        "field_gain": 1.0 - near_goal_weight,
        "W_track": 300.0,
        "W_goal": 0.5 + 25.0 * near_goal_weight,
        "W_terminal_goal": 10.0 + 200.0 * near_goal_weight,
        "robot_radius": float(robot_radius), "rho_geofence": 2.0e7,
        "R_v": 0.01, "R_w": 0.01, "R_dv": 0.05, "R_dw": 0.10,
    }
