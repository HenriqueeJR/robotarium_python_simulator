from __future__ import annotations

import numpy as np
from shapely import intersects_xy
from shapely.geometry import Polygon


def gerar_gvf_lbm(corredores, rotatorias, conexoes, pt_target, map_bounds, res):
    x_vec = np.arange(map_bounds[0], map_bounds[1] + res * 0.5, res)
    y_vec = np.arange(map_bounds[2], map_bounds[3] + res * 0.5, res)
    X_grid, Y_grid = np.meshgrid(x_vec, y_vec)
    obstacles = np.ones(X_grid.shape, dtype=bool)
    force_x = np.zeros_like(X_grid)
    force_y = np.zeros_like(Y_grid)
    force_count = np.zeros_like(X_grid)
    direction_x = np.zeros_like(X_grid)
    direction_y = np.zeros_like(Y_grid)

    for corridor in corredores:
        mask = intersects_xy(Polygon(corridor["v"]), X_grid, Y_grid)
        obstacles[mask] = False
        force_x[mask] += corridor["dir"][0]
        force_y[mask] += corridor["dir"][1]
        force_count[mask] += 1
    valid = force_count > 0
    direction_x[valid] = force_x[valid] / force_count[valid]
    direction_y[valid] = force_y[valid] / force_count[valid]

    theta_poly = np.linspace(0.0, 2.0 * np.pi, 32)
    for roundabout in rotatorias:
        cx, cy, outer_radius, inner_radius = roundabout[:4]
        radius_squared = (X_grid - cx) ** 2 + (Y_grid - cy) ** 2
        outer_mask = radius_squared <= outer_radius**2
        inner_mask = radius_squared <= inner_radius**2
        obstacles[outer_mask] = False
        obstacles[inner_mask] = True
        fluid_ring = outer_mask & ~obstacles
        direction_x[fluid_ring] = -roundabout[4] * (Y_grid[fluid_ring] - cy)
        direction_y[fluid_ring] = roundabout[4] * (X_grid[fluid_ring] - cx)

    for incoming_index, outgoing_index, roundabout_index in conexoes:
        incoming = Polygon(corredores[incoming_index - 1]["v"])
        outgoing = Polygon(corredores[outgoing_index - 1]["v"])
        roundabout = rotatorias[roundabout_index - 1]
        cx, cy, outer_radius = roundabout[:3]
        roundabout_polygon = Polygon(np.column_stack((
            cx + outer_radius * np.cos(theta_poly),
            cy + outer_radius * np.sin(theta_poly),
        )))
        incoming_center = roundabout_polygon.intersection(incoming).centroid
        outgoing_center = roundabout_polygon.intersection(outgoing).centroid
        angle_in = np.arctan2(incoming_center.y - cy, incoming_center.x - cx)
        angle_out = np.arctan2(outgoing_center.y - cy, outgoing_center.x - cx)
        delta_travel = np.mod(angle_out - angle_in, 2.0 * np.pi)
        middle_block = np.mod(angle_out + (2.0 * np.pi - delta_travel) / 2.0,
                              2.0 * np.pi)
        half_width = max(0.0, (2.0 * np.pi - delta_travel) / 2.0 - 0.45)
        angle_grid = np.arctan2(Y_grid - cy, X_grid - cx)
        angle_difference = np.mod(angle_grid - middle_block + np.pi,
                                   2.0 * np.pi) - np.pi
        wedge = np.abs(angle_difference) <= half_width
        mask_roundabout = (X_grid - cx) ** 2 + (Y_grid - cy) ** 2 <= outer_radius**2
        obstacles[mask_roundabout & wedge] = True

    obstacles[[0, -1], :] = True
    obstacles[:, [0, -1]] = True
    norm_direction = np.hypot(direction_x, direction_y) + 1e-10
    force_x_base = 0.0015 * direction_x / norm_direction
    force_y_base = 0.0015 * direction_y / norm_direction
    force_x_base[obstacles] = 0.0
    force_y_base[obstacles] = 0.0
    sink = (((X_grid - pt_target[0]) ** 2 + (Y_grid - pt_target[1]) ** 2)
            <= 0.15**2) & ~obstacles

    tau = 0.8
    weights = np.array([4 / 9, *([1 / 9] * 4), *([1 / 36] * 4)])
    cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    rho = np.ones_like(X_grid)
    ux = np.zeros_like(X_grid)
    uy = np.zeros_like(Y_grid)
    distributions = np.empty((*X_grid.shape, 9), dtype=float)
    for direction in range(9):
        distributions[:, :, direction] = weights[direction] * rho

    for _ in range(3000):
        rho = np.sum(distributions, axis=2)
        ux.fill(0.0)
        uy.fill(0.0)
        for direction in range(9):
            ux += cx[direction] * distributions[:, :, direction]
            uy += cy[direction] * distributions[:, :, direction]
        fluid = ~obstacles
        fx = force_x_base - 0.02 * ux
        fy = force_y_base - 0.02 * uy
        ux[fluid] = (ux[fluid] + 0.5 * fx[fluid]) / rho[fluid]
        uy[fluid] = (uy[fluid] + 0.5 * fy[fluid]) / rho[fluid]
        ux[obstacles] = 0.0
        uy[obstacles] = 0.0

        outgoing = np.empty_like(distributions)
        velocity_squared = ux**2 + uy**2
        velocity_force = ux * fx + uy * fy
        for direction in range(9):
            c_dot_u = cx[direction] * ux + cy[direction] * uy
            c_dot_force = cx[direction] * fx + cy[direction] * fy
            equilibrium = weights[direction] * rho * (
                1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u**2 - 1.5 * velocity_squared
            )
            forcing = (1.0 - 0.5 / tau) * weights[direction] * (
                3.0 * c_dot_force + 9.0 * c_dot_u * c_dot_force
                - 3.0 * velocity_force
            )
            outgoing[:, :, direction] = (
                distributions[:, :, direction]
                - (distributions[:, :, direction] - equilibrium) / tau
                + forcing
            )

        streamed = np.empty_like(distributions)
        for direction in range(9):
            streamed[:, :, direction] = np.roll(
                outgoing[:, :, direction],
                shift=(cy[direction], cx[direction]),
                axis=(0, 1),
            )
        distributions = streamed.copy()
        for direction in range(1, 9):
            current = distributions[:, :, direction]
            opposite_values = streamed[:, :, opposite[direction]]
            current[obstacles] = opposite_values[obstacles]
        for direction in range(9):
            distributions[:, :, direction][sink] = weights[direction] * 0.15

    return X_grid, Y_grid, ux, uy
