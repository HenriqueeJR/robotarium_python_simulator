from __future__ import annotations

import numpy as np

from .dynamics import interpolate_vector_field, unicycle_step
from .geofence import calc_point_p2s_penalty_dyn
from ._numba_dcbf import KernelParams, build_local_dcbf_kernel


def build_local_dcbf(U, params, field, return_details=True):
    kernel_params = KernelParams(
        int(params["N"]), float(params["Ts"]),
        np.ascontiguousarray(params["x0"], dtype=np.float64),
        np.ascontiguousarray(params["neighbor_positions"], dtype=np.float64),
        np.ascontiguousarray(params["u_previous"], dtype=np.float64),
        np.ascontiguousarray(params["target"], dtype=np.float64),
        float(params["d_safe"]), float(params["gamma_cbf"]),
        float(params["rho_cbf"]),
        np.ascontiguousarray(params["rotatorias"], dtype=np.float64),
        int(params["N_blocks"]),
        np.ascontiguousarray(params["blocks_params"], dtype=np.float64),
        float(params["h_p2s"]), float(params["eta_safe"]),
        float(params["gamma_safe"]), float(params["eta_obs"]),
        float(params["gamma_obs"]), float(params["field_gain"]),
        float(params["W_track"]), float(params["W_goal"]),
        float(params["W_terminal_goal"]), float(params["robot_radius"]),
        float(params["rho_geofence"]), float(params["R_v"]),
        float(params["R_w"]), float(params["R_dv"]), float(params["R_dw"]),
    )
    result = build_local_dcbf_kernel(
        np.ascontiguousarray(U, dtype=np.float64), kernel_params,
        np.ascontiguousarray(field["x_vec"], dtype=np.float64),
        np.ascontiguousarray(field["y_vec"], dtype=np.float64),
        np.ascontiguousarray(field["ux"], dtype=np.float64),
        np.ascontiguousarray(field["uy"], dtype=np.float64),
        bool(return_details),
    )
    if not return_details:
        return result[0]
    cost, gradient, predicted, max_dcbf, max_geofence, min_distance = result
    diagnostics = {
        "max_dcbf_violation": max_dcbf,
        "max_geofence_violation": max_geofence,
        "min_predicted_distance": min_distance,
    }
    return cost, gradient, predicted, diagnostics


def _build_local_dcbf_python(U, params, field, return_details=True):
    N = params["N"]
    Ts = params["Ts"]
    controls = np.asarray(U, dtype=float).reshape(2, N, order="F")
    predicted = np.zeros((3, N + 1), dtype=float)
    predicted[:, 0] = params["x0"]
    stage_x = np.zeros((3, N))
    stage_u = np.zeros((2, N))
    cbf_x = np.zeros((3, N))
    cbf_xnext = np.zeros((3, N))
    cost = 0.0
    max_dcbf_violation = 0.0
    max_geofence_violation = 0.0
    min_distance = float("inf")
    neighbors = np.asarray(params["neighbor_positions"])
    n_neighbors = neighbors.shape[2] if neighbors.ndim == 3 else 0

    for step in range(N):
        state = predicted[:, step]
        velocity, omega = controls[:, step]
        theta = state[2]
        vx, vy, dvx_dx, dvx_dy, dvy_dx, dvy_dy = interpolate_vector_field(field, state[:2])
        vx *= params["field_gain"]
        vy *= params["field_gain"]
        dvx_dx *= params["field_gain"]
        dvx_dy *= params["field_gain"]
        dvy_dx *= params["field_gain"]
        dvy_dy *= params["field_gain"]
        error_x = velocity * np.cos(theta) - vx
        error_y = velocity * np.sin(theta) - vy
        tracking_cost = params["W_track"] * (error_x**2 + error_y**2)
        goal_error = state[:2] - params["target"]
        goal_cost = params["W_goal"] * np.dot(goal_error, goal_error)
        next_state = unicycle_step(state, controls[:, step], Ts)
        predicted[:, step + 1] = next_state

        geo_n, geo_grad_n = calc_point_p2s_penalty_dyn(
            state[:2], params["blocks_params"], params["rotatorias"],
            params["h_p2s"], params["robot_radius"],
        )
        geo_next, geo_grad_next = calc_point_p2s_penalty_dyn(
            next_state[:2], params["blocks_params"], params["rotatorias"],
            params["h_p2s"], params["robot_radius"],
        )
        h_geo_n = -geo_n
        h_geo_next = -geo_next
        g_geo = (1.0 - params["gamma_safe"]) * h_geo_n - h_geo_next
        geofence_cost = params["eta_safe"] * max(0.0, g_geo) ** 2
        max_geofence_violation = max(max_geofence_violation, max(0.0, g_geo))
        if g_geo > 0:
            multiplier = 2.0 * params["eta_safe"] * g_geo
            cbf_x[:2, step] -= multiplier * (1.0 - params["gamma_safe"]) * geo_grad_n
            cbf_xnext[:2, step] += multiplier * geo_grad_next

        obstacle_cost = 0.0
        for roundabout in params["rotatorias"]:
            center = roundabout[:2]
            safe_radius = roundabout[3] + params["robot_radius"]
            delta_n = state[:2] - center
            delta_next = next_state[:2] - center
            violation_n = max(0.0, safe_radius**2 - np.dot(delta_n, delta_n))
            violation_next = max(0.0, safe_radius**2 - np.dot(delta_next, delta_next))
            h_n = -(violation_n**2)
            h_next = -(violation_next**2)
            g_obs = (1.0 - params["gamma_obs"]) * h_n - h_next
            obstacle_cost += params["eta_obs"] * max(0.0, g_obs) ** 2
            if g_obs > 0:
                multiplier = 2.0 * params["eta_obs"] * g_obs
                grad_h_n = (4.0 * violation_n * delta_n if violation_n > 0
                            else np.zeros(2))
                grad_h_next = (4.0 * violation_next * delta_next
                               if violation_next > 0 else np.zeros(2))
                cbf_x[:2, step] += multiplier * (1.0 - params["gamma_obs"]) * grad_h_n
                cbf_xnext[:2, step] -= multiplier * grad_h_next

        input_cost = params["R_v"] * velocity**2 + params["R_w"] * omega**2
        cost += tracking_cost + goal_cost + geofence_cost + obstacle_cost + input_cost
        dex = np.array([-dvx_dx, -dvx_dy, -velocity * np.sin(theta)])
        dey = np.array([-dvy_dx, -dvy_dy, velocity * np.cos(theta)])
        stage_x[:, step] = 2.0 * params["W_track"] * (
            error_x * dex + error_y * dey
        )
        stage_x[:2, step] += 2.0 * params["W_goal"] * goal_error
        stage_u[:, step] = [
            2.0 * params["R_v"] * velocity
            + 2.0 * params["W_track"] * (
                error_x * np.cos(theta) + error_y * np.sin(theta)
            ),
            2.0 * params["R_w"] * omega,
        ]

        for neighbor in range(n_neighbors):
            neighbor_n = neighbors[:, step, neighbor]
            neighbor_next = neighbors[:, step + 1, neighbor]
            delta_n = state[:2] - neighbor_n
            delta_next = next_state[:2] - neighbor_next
            min_distance = min(min_distance, np.linalg.norm(delta_n),
                               np.linalg.norm(delta_next))
            h_n = np.dot(delta_n, delta_n) - params["d_safe"] ** 2
            h_next = np.dot(delta_next, delta_next) - params["d_safe"] ** 2
            violation = (1.0 - params["gamma_cbf"]) * h_n - h_next
            positive = max(0.0, violation)
            cost += params["rho_cbf"] * positive**2
            max_dcbf_violation = max(max_dcbf_violation, positive)
            if violation > 0:
                multiplier = 2.0 * params["rho_cbf"] * violation
                cbf_x[:2, step] += multiplier * (1.0 - params["gamma_cbf"]) * 2.0 * delta_n
                cbf_xnext[:2, step] -= multiplier * 2.0 * delta_next

    terminal = predicted[:, -1]
    terminal_error = terminal[:2] - params["target"]
    terminal_geo, terminal_geo_grad = calc_point_p2s_penalty_dyn(
        terminal[:2], params["blocks_params"], params["rotatorias"],
        params["h_p2s"], params["robot_radius"],
    )
    cost += (params["W_terminal_goal"] * np.dot(terminal_error, terminal_error)
             + params["rho_geofence"] * terminal_geo**2)
    terminal_gradient = np.r_[
        2.0 * params["W_terminal_goal"] * terminal_error
        + 2.0 * params["rho_geofence"] * terminal_geo * terminal_geo_grad,
        0.0,
    ]

    smooth_gradient = np.zeros((2, N))
    previous_control = params["u_previous"].copy()
    for step in range(N):
        delta = controls[:, step] - previous_control
        cost += params["R_dv"] * delta[0] ** 2 + params["R_dw"] * delta[1] ** 2
        local_gradient = np.array([
            2.0 * params["R_dv"] * delta[0],
            2.0 * params["R_dw"] * delta[1],
        ])
        smooth_gradient[:, step] += local_gradient
        if step > 0:
            smooth_gradient[:, step - 1] -= local_gradient
        previous_control = controls[:, step]

    diagnostics = {
        "max_dcbf_violation": max_dcbf_violation,
        "max_geofence_violation": max_geofence_violation,
        "min_predicted_distance": min_distance,
    }
    if not return_details:
        return cost

    gradient_matrix = np.zeros((2, N))
    costate = terminal_gradient
    for step in range(N - 1, -1, -1):
        state = predicted[:, step]
        velocity = controls[0, step]
        theta = state[2]
        Fx = np.array([
            [1.0, 0.0, -Ts * velocity * np.sin(theta)],
            [0.0, 1.0, Ts * velocity * np.cos(theta)],
            [0.0, 0.0, 1.0],
        ])
        Fu = np.array([
            [Ts * np.cos(theta), 0.0],
            [Ts * np.sin(theta), 0.0],
            [0.0, Ts],
        ])
        transition_costate = costate + cbf_xnext[:, step]
        gradient_matrix[:, step] = (
            stage_u[:, step] + Fu.T @ transition_costate + smooth_gradient[:, step]
        )
        costate = stage_x[:, step] + cbf_x[:, step] + Fx.T @ transition_costate
    return cost, gradient_matrix.reshape(-1, order="F"), predicted, diagnostics
