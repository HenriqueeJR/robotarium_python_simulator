from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit


class KernelParams(NamedTuple):
    N: int
    Ts: float
    x0: np.ndarray
    neighbor_positions: np.ndarray
    u_previous: np.ndarray
    target: np.ndarray
    d_safe: float
    gamma_cbf: float
    rho_cbf: float
    rotatorias: np.ndarray
    N_blocks: int
    blocks_params: np.ndarray
    h_p2s: float
    eta_safe: float
    gamma_safe: float
    eta_obs: float
    gamma_obs: float
    field_gain: float
    W_track: float
    W_goal: float
    W_terminal_goal: float
    robot_radius: float
    rho_geofence: float
    R_v: float
    R_w: float
    R_dv: float
    R_dw: float


@njit(cache=True)
def _phi_func(s, h):
    if s >= 0.0:
        return (s**3) / (2.0 * (s + h)), (s**2 * (2.0 * s + 3.0 * h)) / (2.0 * (s + h) ** 2)
    return 0.0, 0.0


@njit(cache=True)
def _get_single_block_p2s(point, block, h):
    v1, g1 = _phi_func(block[0] - point[0], h)
    v2, g2 = _phi_func(point[0] - block[1], h)
    v3, g3 = _phi_func(block[2] - point[1], h)
    v4, g4 = _phi_func(point[1] - block[3], h)
    value = 0.25 * (v1 + v2 + v3 + v4)
    gradient = np.empty(2)
    gradient[0] = 0.25 * (-g1 + g2)
    gradient[1] = 0.25 * (-g3 + g4)
    return value, gradient


@njit(cache=True)
def _smooth_min_func(a, b, gradient_a, gradient_b):
    difference = a - b
    delta = np.sqrt(difference * difference + 1e-10)
    value = 0.5 * (a + b - delta)
    gradient = 0.5 * (
        gradient_a + gradient_b - (difference / delta) * (gradient_a - gradient_b)
    )
    return value, gradient


@njit(cache=True)
def _get_roundabout_p2s(point, center_x, center_y, outer_radius, robot_radius):
    dx = point[0] - center_x
    dy = point[1] - center_y
    violation = max(0.0, dx * dx + dy * dy - (outer_radius - robot_radius) ** 2)
    value = violation * violation
    gradient = np.zeros(2)
    if violation > 0.0:
        gradient[0] = 4.0 * violation * dx
        gradient[1] = 4.0 * violation * dy
    return value, gradient


@njit(cache=True)
def _calc_point_p2s_penalty(point, blocks, n_blocks, roundabouts, h, robot_radius):
    value = 0.0
    gradient = np.zeros(2)
    first = True
    for index in range(roundabouts.shape[0]):
        candidate, candidate_gradient = _get_roundabout_p2s(
            point, roundabouts[index, 0], roundabouts[index, 1],
            roundabouts[index, 2], robot_radius,
        )
        if first:
            value = candidate
            gradient = candidate_gradient
            first = False
        else:
            value, gradient = _smooth_min_func(
                value, candidate, gradient, candidate_gradient
            )
    for index in range(n_blocks):
        candidate, candidate_gradient = _get_single_block_p2s(
            point, blocks[4 * index:4 * index + 4], h
        )
        if first:
            value = candidate
            gradient = candidate_gradient
            first = False
        else:
            value, gradient = _smooth_min_func(
                value, candidate, gradient, candidate_gradient
            )
    return value, gradient


@njit(cache=True)
def _interpolate_vector_field(x_vec, y_vec, ux_grid, uy_grid, position):
    dx = x_vec[1] - x_vec[0]
    dy = y_vec[1] - y_vec[0]
    x = min(max(position[0], x_vec[0]), x_vec[-1] - 1.01 * dx)
    y = min(max(position[1], y_vec[0]), y_vec[-1] - 1.01 * dy)
    column = min(max(int(np.floor((x - x_vec[0]) / dx)), 0), x_vec.size - 2)
    row = min(max(int(np.floor((y - y_vec[0]) / dy)), 0), y_vec.size - 2)
    tx = (x - x_vec[column]) / dx
    ty = (y - y_vec[row]) / dy

    ux00 = ux_grid[row, column]
    ux10 = ux_grid[row, column + 1]
    ux01 = ux_grid[row + 1, column]
    ux11 = ux_grid[row + 1, column + 1]
    uy00 = uy_grid[row, column]
    uy10 = uy_grid[row, column + 1]
    uy01 = uy_grid[row + 1, column]
    uy11 = uy_grid[row + 1, column + 1]

    vx = (1.0 - tx) * (1.0 - ty) * ux00 + tx * (1.0 - ty) * ux10 + (1.0 - tx) * ty * ux01 + tx * ty * ux11
    vy = (1.0 - tx) * (1.0 - ty) * uy00 + tx * (1.0 - ty) * uy10 + (1.0 - tx) * ty * uy01 + tx * ty * uy11
    dvx_dx = ((1.0 - ty) * (ux10 - ux00) + ty * (ux11 - ux01)) / dx
    dvx_dy = ((1.0 - tx) * (ux01 - ux00) + tx * (ux11 - ux10)) / dy
    dvy_dx = ((1.0 - ty) * (uy10 - uy00) + ty * (uy11 - uy01)) / dx
    dvy_dy = ((1.0 - tx) * (uy01 - uy00) + tx * (uy11 - uy10)) / dy
    return vx, vy, dvx_dx, dvx_dy, dvy_dx, dvy_dy


@njit(cache=True)
def build_local_dcbf_kernel(
    U, params, x_vec, y_vec, ux_grid, uy_grid, calculate_gradient=True,
):
    N = params.N
    Ts = params.Ts
    controls = np.empty((2, N))
    for index in range(N):
        controls[0, index] = U[2 * index]
        controls[1, index] = U[2 * index + 1]

    predicted = np.zeros((3, N + 1))
    predicted[:, 0] = params.x0
    stage_x = np.zeros((3, N))
    stage_u = np.zeros((2, N))
    cbf_x = np.zeros((3, N))
    cbf_xnext = np.zeros((3, N))
    cost = 0.0
    max_dcbf_violation = 0.0
    max_geofence_violation = 0.0
    min_distance = np.inf
    n_neighbors = params.neighbor_positions.shape[2]

    for step in range(N):
        state = predicted[:, step]
        velocity = controls[0, step]
        omega = controls[1, step]
        theta = state[2]
        vx, vy, dvx_dx, dvx_dy, dvy_dx, dvy_dy = _interpolate_vector_field(
            x_vec, y_vec, ux_grid, uy_grid, state[:2]
        )
        vx *= params.field_gain
        vy *= params.field_gain
        dvx_dx *= params.field_gain
        dvx_dy *= params.field_gain
        dvy_dx *= params.field_gain
        dvy_dy *= params.field_gain
        error_x = velocity * np.cos(theta) - vx
        error_y = velocity * np.sin(theta) - vy
        tracking_cost = params.W_track * (error_x**2 + error_y**2)
        goal_error = state[:2] - params.target
        goal_cost = params.W_goal * (goal_error[0]**2 + goal_error[1]**2)

        next_state = np.empty(3)
        next_state[0] = state[0] + Ts * velocity * np.cos(theta)
        next_state[1] = state[1] + Ts * velocity * np.sin(theta)
        next_state[2] = state[2] + Ts * omega
        predicted[:, step + 1] = next_state

        geo_n, geo_grad_n = _calc_point_p2s_penalty(
            state[:2], params.blocks_params, params.N_blocks, params.rotatorias,
            params.h_p2s, params.robot_radius,
        )
        geo_next, geo_grad_next = _calc_point_p2s_penalty(
            next_state[:2], params.blocks_params, params.N_blocks, params.rotatorias,
            params.h_p2s, params.robot_radius,
        )
        g_geo = geo_next - (1.0 - params.gamma_safe) * geo_n
        geofence_positive = max(0.0, g_geo)
        geofence_cost = params.eta_safe * geofence_positive**2
        max_geofence_violation = max(max_geofence_violation, geofence_positive)
        if calculate_gradient and g_geo > 0.0:
            multiplier = 2.0 * params.eta_safe * g_geo
            cbf_x[:2, step] -= multiplier * (1.0 - params.gamma_safe) * geo_grad_n
            cbf_xnext[:2, step] += multiplier * geo_grad_next

        obstacle_cost = 0.0
        for roundabout_index in range(params.rotatorias.shape[0]):
            center_x = params.rotatorias[roundabout_index, 0]
            center_y = params.rotatorias[roundabout_index, 1]
            safe_radius = params.rotatorias[roundabout_index, 3] + params.robot_radius
            dx_n = state[0] - center_x
            dy_n = state[1] - center_y
            dx_next = next_state[0] - center_x
            dy_next = next_state[1] - center_y
            violation_n = max(0.0, safe_radius**2 - dx_n**2 - dy_n**2)
            violation_next = max(0.0, safe_radius**2 - dx_next**2 - dy_next**2)
            h_n = -(violation_n**2)
            h_next = -(violation_next**2)
            g_obs = (1.0 - params.gamma_obs) * h_n - h_next
            obstacle_positive = max(0.0, g_obs)
            obstacle_cost += params.eta_obs * obstacle_positive**2
            if calculate_gradient and g_obs > 0.0:
                multiplier = 2.0 * params.eta_obs * g_obs
                grad_h_n = np.zeros(2)
                grad_h_next = np.zeros(2)
                if violation_n > 0.0:
                    grad_h_n[0] = 4.0 * violation_n * dx_n
                    grad_h_n[1] = 4.0 * violation_n * dy_n
                if violation_next > 0.0:
                    grad_h_next[0] = 4.0 * violation_next * dx_next
                    grad_h_next[1] = 4.0 * violation_next * dy_next
                cbf_x[:2, step] += multiplier * (1.0 - params.gamma_obs) * grad_h_n
                cbf_xnext[:2, step] -= multiplier * grad_h_next

        cost += (tracking_cost + goal_cost + geofence_cost + obstacle_cost
                 + params.R_v * velocity**2 + params.R_w * omega**2)

        if calculate_gradient:
            dex_dx = -dvx_dx
            dex_dy = -dvx_dy
            dex_dtheta = -velocity * np.sin(theta)
            dey_dx = -dvy_dx
            dey_dy = -dvy_dy
            dey_dtheta = velocity * np.cos(theta)
            stage_x[0, step] = 2.0 * params.W_track * (error_x * dex_dx + error_y * dey_dx)
            stage_x[1, step] = 2.0 * params.W_track * (error_x * dex_dy + error_y * dey_dy)
            stage_x[2, step] = 2.0 * params.W_track * (error_x * dex_dtheta + error_y * dey_dtheta)
            stage_x[:2, step] += 2.0 * params.W_goal * goal_error
            stage_u[0, step] = (
                2.0 * params.R_v * velocity
                + 2.0 * params.W_track * (error_x * np.cos(theta) + error_y * np.sin(theta))
            )
            stage_u[1, step] = 2.0 * params.R_w * omega

        for neighbor in range(n_neighbors):
            delta_n_x = state[0] - params.neighbor_positions[0, step, neighbor]
            delta_n_y = state[1] - params.neighbor_positions[1, step, neighbor]
            delta_next_x = next_state[0] - params.neighbor_positions[0, step + 1, neighbor]
            delta_next_y = next_state[1] - params.neighbor_positions[1, step + 1, neighbor]
            dist_n = np.sqrt(delta_n_x**2 + delta_n_y**2)
            dist_next = np.sqrt(delta_next_x**2 + delta_next_y**2)
            min_distance = min(min_distance, dist_n, dist_next)
            h_n = delta_n_x**2 + delta_n_y**2 - params.d_safe**2
            h_next = delta_next_x**2 + delta_next_y**2 - params.d_safe**2
            violation = (1.0 - params.gamma_cbf) * h_n - h_next
            positive = max(0.0, violation)
            cost += params.rho_cbf * positive**2
            max_dcbf_violation = max(max_dcbf_violation, positive)
            if calculate_gradient and violation > 0.0:
                multiplier = 2.0 * params.rho_cbf * violation
                cbf_x[0, step] += multiplier * (1.0 - params.gamma_cbf) * 2.0 * delta_n_x
                cbf_x[1, step] += multiplier * (1.0 - params.gamma_cbf) * 2.0 * delta_n_y
                cbf_xnext[0, step] -= multiplier * 2.0 * delta_next_x
                cbf_xnext[1, step] -= multiplier * 2.0 * delta_next_y

    terminal = predicted[:, N]
    terminal_error_x = terminal[0] - params.target[0]
    terminal_error_y = terminal[1] - params.target[1]
    terminal_geo, terminal_geo_grad = _calc_point_p2s_penalty(
        terminal[:2], params.blocks_params, params.N_blocks, params.rotatorias,
        params.h_p2s, params.robot_radius,
    )
    cost += (
        params.W_terminal_goal * (terminal_error_x**2 + terminal_error_y**2)
        + params.rho_geofence * terminal_geo**2
    )

    smooth_gradient = np.zeros((2, N))
    previous_u = params.u_previous.copy()
    for step in range(N):
        delta_v = controls[0, step] - previous_u[0]
        delta_w = controls[1, step] - previous_u[1]
        cost += params.R_dv * delta_v**2 + params.R_dw * delta_w**2
        if calculate_gradient:
            gradient_v = 2.0 * params.R_dv * delta_v
            gradient_w = 2.0 * params.R_dw * delta_w
            smooth_gradient[0, step] += gradient_v
            smooth_gradient[1, step] += gradient_w
            if step > 0:
                smooth_gradient[0, step - 1] -= gradient_v
                smooth_gradient[1, step - 1] -= gradient_w
        previous_u[0] = controls[0, step]
        previous_u[1] = controls[1, step]

    if not calculate_gradient:
        return cost, np.zeros(2 * N), predicted, max_dcbf_violation, max_geofence_violation, min_distance

    terminal_gradient = np.zeros(3)
    terminal_gradient[0] = 2.0 * params.W_terminal_goal * terminal_error_x + 2.0 * params.rho_geofence * terminal_geo * terminal_geo_grad[0]
    terminal_gradient[1] = 2.0 * params.W_terminal_goal * terminal_error_y + 2.0 * params.rho_geofence * terminal_geo * terminal_geo_grad[1]

    gradient = np.zeros(2 * N)
    costate = terminal_gradient
    for step in range(N - 1, -1, -1):
        state = predicted[:, step]
        velocity = controls[0, step]
        theta = state[2]
        transition_costate = costate + cbf_xnext[:, step]
        gradient[2 * step] = (
            stage_u[0, step] + Ts * np.cos(theta) * transition_costate[0]
            + Ts * np.sin(theta) * transition_costate[1]
            + smooth_gradient[0, step]
        )
        gradient[2 * step + 1] = (
            stage_u[1, step] + Ts * transition_costate[2]
            + smooth_gradient[1, step]
        )
        costate = np.empty(3)
        costate[0] = stage_x[0, step] + cbf_x[0, step] + transition_costate[0]
        costate[1] = stage_x[1, step] + cbf_x[1, step] + transition_costate[1]
        costate[2] = (
            stage_x[2, step] + cbf_x[2, step]
            - Ts * velocity * np.sin(theta) * transition_costate[0]
            + Ts * velocity * np.cos(theta) * transition_costate[1]
            + transition_costate[2]
        )

    return cost, gradient, predicted, max_dcbf_violation, max_geofence_violation, min_distance
