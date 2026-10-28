from __future__ import annotations

import numpy as np


def pack_params_1D(
    x0, neighbor_positions, u_previous, target, N, Ts, d_safe, gamma_cbf,
    rho_cbf, robot_radius, corredores, rotatorias, agent_id,
):
    """Pack parameters into the fixed 350-element MATLAB-Coder layout."""
    neighbors = np.asarray(neighbor_positions, dtype=float)
    n_neighbors = neighbors.shape[2] if neighbors.ndim == 3 else 0
    if n_neighbors > 3:
        raise ValueError("At most three neighbors are supported.")
    if len(corredores) > 20 or len(rotatorias) > 5:
        raise ValueError("The packed layout supports at most 20 blocks and 5 roundabouts.")

    packed = np.zeros(350, dtype=float)
    packed[0:3] = np.asarray(x0).reshape(3)
    packed[3:5] = np.asarray(target).reshape(2)
    packed[5:7] = np.asarray(u_previous).reshape(2)
    packed[7:13] = [N, Ts, d_safe, gamma_cbf, rho_cbf, robot_radius]
    distance = np.linalg.norm(packed[0:2] - packed[3:5])
    near_goal = np.exp(-(distance / 0.25) ** 2)
    packed[13:22] = [
        1.0 - near_goal, 300.0, 0.5 + 25.0 * near_goal,
        10.0 + 200.0 * near_goal, 2.0e7, 0.01, 0.01, 0.05, 0.10,
    ]
    packed[22:27] = [1.0e7, 1.0e7, 0.5, 0.5, 0.05]
    packed[27:31] = [len(corredores), len(rotatorias), n_neighbors, agent_id]

    for index, corridor in enumerate(corredores):
        vertices = np.asarray(corridor["v"])
        packed[31 + 4 * index:35 + 4 * index] = [
            vertices[:, 0].min() + robot_radius,
            vertices[:, 0].max() - robot_radius,
            vertices[:, 1].min() + robot_radius,
            vertices[:, 1].max() - robot_radius,
        ]
    for index, roundabout in enumerate(np.asarray(rotatorias)):
        packed[111 + 4 * index:115 + 4 * index] = roundabout[:4]

    for neighbor in range(n_neighbors):
        base = 131 + neighbor * 2 * (N + 1)
        packed[base:base + 2 * (N + 1)] = neighbors[:, :, neighbor].T.reshape(-1)
    return packed


def unpack_neighbors(packed, N, n_neighbors):
    packed = np.asarray(packed, dtype=float).reshape(-1)
    if packed.size < 317 or n_neighbors > 3:
        raise ValueError("Packed vector is too short or neighbor count is invalid.")
    flat = packed[131:317]
    decoded = np.zeros((2, N + 1, n_neighbors), dtype=float)
    for neighbor in range(n_neighbors):
        start = neighbor * 2 * (N + 1)
        decoded[:, :, neighbor] = flat[
            start:start + 2 * (N + 1)
        ].reshape(N + 1, 2).T
    return decoded
