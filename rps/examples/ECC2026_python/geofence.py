from __future__ import annotations

import numpy as np


def phi_func(s, h):
    if s >= 0:
        value = s**3 / (2.0 * (s + h))
        gradient = s**2 * (2.0 * s + 3.0 * h) / (2.0 * (s + h) ** 2)
        return value, gradient
    return 0.0, 0.0


def get_single_block_p2s(point, block, h):
    x, y = point
    x_min, x_max, y_min, y_max = block
    values = [
        phi_func(x_min - x, h), phi_func(x - x_max, h),
        phi_func(y_min - y, h), phi_func(y - y_max, h),
    ]
    penalty = 0.25 * sum(value for value, _ in values)
    gradient = 0.25 * np.array([
        -values[0][1] + values[1][1],
        -values[2][1] + values[3][1],
    ])
    return penalty, gradient


def smooth_min_func(a, b, gradient_a, gradient_b):
    difference = a - b
    delta = np.sqrt(difference**2 + 1e-10)
    value = 0.5 * (a + b - delta)
    gradient = 0.5 * (
        gradient_a + gradient_b
        - (difference / delta) * (gradient_a - gradient_b)
    )
    return value, gradient


def get_roundabout_p2s(point, center, outer_radius, robot_radius):
    delta = np.asarray(point) - np.asarray(center)
    violation = max(0.0, np.dot(delta, delta) - (outer_radius - robot_radius) ** 2)
    penalty = violation**2
    gradient = 4.0 * violation * delta if violation > 0 else np.zeros(2)
    return penalty, gradient


def calc_point_p2s_penalty_dyn(point, blocks, rotatorias, h, robot_radius):
    penalties = []
    for roundabout in rotatorias:
        penalties.append(get_roundabout_p2s(
            point, roundabout[:2], roundabout[2], robot_radius
        ))
    for block in np.asarray(blocks).reshape(-1, 4):
        penalties.append(get_single_block_p2s(point, block, h))
    if not penalties:
        return 0.0, np.zeros(2)
    total, gradient = penalties[0]
    for penalty, penalty_gradient in penalties[1:]:
        total, gradient = smooth_min_func(
            total, penalty, gradient, penalty_gradient
        )
    return total, gradient


def build_route_geofence_grid(X_grid, Y_grid, route_boxes,
                              route_roundabouts, robot_radius):
    geofence = np.full(np.shape(X_grid), -np.inf)
    for bounds in np.asarray(route_boxes).reshape(4, -1).T:
        x_min, x_max, y_min, y_max = bounds
        h_box = np.minimum.reduce((
            X_grid - (x_min + robot_radius),
            (x_max - robot_radius) - X_grid,
            Y_grid - (y_min + robot_radius),
            (y_max - robot_radius) - Y_grid,
        ))
        geofence = np.maximum(geofence, h_box)
    for roundabout in np.asarray(route_roundabouts).reshape(-1, 5):
        cx, cy, outer_radius = roundabout[:3]
        safe_radius = outer_radius - robot_radius
        radius_squared = (X_grid - cx) ** 2 + (Y_grid - cy) ** 2
        h_outer = (safe_radius**2 - radius_squared) / (2.0 * safe_radius)
        geofence = np.maximum(geofence, h_outer)
    for roundabout in np.asarray(route_roundabouts).reshape(-1, 5):
        cx, cy, _, inner_radius = roundabout[:4]
        safe_radius = inner_radius + robot_radius
        radius_squared = (X_grid - cx) ** 2 + (Y_grid - cy) ** 2
        h_inner = (radius_squared - safe_radius**2) / (2.0 * safe_radius)
        geofence = np.minimum(geofence, h_inner)
    return geofence
