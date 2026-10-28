from __future__ import annotations

import numpy as np


def unicycle_step(state, control, Ts):
    x, y, theta = state
    v, omega = control
    return np.array([
        x + Ts * v * np.cos(theta),
        y + Ts * v * np.sin(theta),
        theta + Ts * omega,
    ])


def rollout_unicycle(x0, U, Ts, N):
    controls = np.asarray(U, dtype=float).reshape(2, N)
    states = np.zeros((3, N + 1), dtype=float)
    states[:, 0] = np.asarray(x0).reshape(3)
    for index in range(N):
        states[:, index + 1] = unicycle_step(
            states[:, index], controls[:, index], Ts
        )
    return states


def shift_control_sequence(U, N):
    controls = np.asarray(U, dtype=float).reshape(2, N)
    return np.column_stack((controls[:, 1:], controls[:, -1])).reshape(-1)


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def interpolate_vector_field(field, position):
    x_vec = np.asarray(field["x_vec"], dtype=float).reshape(-1)
    y_vec = np.asarray(field["y_vec"], dtype=float).reshape(-1)
    dx = x_vec[1] - x_vec[0]
    dy = y_vec[1] - y_vec[0]
    x = np.clip(position[0], x_vec[0], x_vec[-1] - 1.01 * dx)
    y = np.clip(position[1], y_vec[0], y_vec[-1] - 1.01 * dy)
    column = int(np.clip(np.floor((x - x_vec[0]) / dx), 0, len(x_vec) - 2))
    row = int(np.clip(np.floor((y - y_vec[0]) / dy), 0, len(y_vec) - 2))
    tx = (x - x_vec[column]) / dx
    ty = (y - y_vec[row]) / dy
    values = {}
    for key in ("ux", "uy"):
        matrix = np.asarray(field[key])
        z00 = matrix[row, column]
        z10 = matrix[row, column + 1]
        z01 = matrix[row + 1, column]
        z11 = matrix[row + 1, column + 1]
        values[key] = (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11
        values[f"d{key}_dx"] = ((1 - ty) * (z10 - z00) + ty * (z11 - z01)) / dx
        values[f"d{key}_dy"] = ((1 - tx) * (z01 - z00) + tx * (z11 - z10)) / dy
    return (values["ux"], values["uy"], values["dux_dx"], values["dux_dy"],
            values["duy_dx"], values["duy_dy"])


def make_field_nominal_plan(x0, field, target, N, Ts, speed_limit,
                            omega_min, omega_max):
    controls = np.zeros((2, N), dtype=float)
    state = np.asarray(x0, dtype=float).reshape(3).copy()
    for index in range(N):
        field_x, field_y, *_ = interpolate_vector_field(field, state[:2])
        distance = np.linalg.norm(np.asarray(target) - state[:2])
        goal_blend = np.exp(-(distance / 0.25) ** 2)
        desired = (1.0 - goal_blend) * np.array([field_x, field_y])
        desired += goal_blend * 0.8 * (np.asarray(target) - state[:2])
        desired_speed = np.linalg.norm(desired)
        if desired_speed > 1e-5:
            desired_heading = np.arctan2(desired[1], desired[0])
            speed = min(speed_limit, desired_speed)
        else:
            desired_heading = state[2]
            speed = 0.0
        omega = np.clip(2.0 * wrap_angle(desired_heading - state[2]),
                        omega_min, omega_max)
        controls[:, index] = [speed, omega]
        state = unicycle_step(state, controls[:, index], Ts)
    return controls.reshape(-1, order="F")
