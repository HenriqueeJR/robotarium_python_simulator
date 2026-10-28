from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
import argparse
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.ticker import ScalarFormatter

from rps.robotarium import Robotarium

from .PanocSolver import PanocSolver
from .build_local_dcbf import build_local_dcbf
from .dynamics import make_field_nominal_plan, rollout_unicycle, shift_control_sequence
from .geofence import build_route_geofence_grid, calc_point_p2s_penalty_dyn
from .gerar_gvf_lbm import gerar_gvf_lbm
from .make_local_params import make_local_params
from .planejar_rota import planejar_rota
from .wrapper_cost import wrapper_cost
from .wrapper_grad import wrapper_grad

_WORKER_FIELDS = None
_WORKER_CORRIDORS = None
_WORKER_ROUNDABOUTS = None
_WORKER_CONFIG = None


def _initialize_controller_workers(fields, corredores, rotatorias, config):
    global _WORKER_FIELDS, _WORKER_CORRIDORS, _WORKER_ROUNDABOUTS, _WORKER_CONFIG
    _WORKER_FIELDS = fields
    _WORKER_CORRIDORS = corredores
    _WORKER_ROUNDABOUTS = rotatorias
    _WORKER_CONFIG = config


def _compute_agent_control(job):
    control_started_ns = time.perf_counter_ns()
    agent, state, neighbor_positions, previous_control, warm_start, solver = job
    config = _WORKER_CONFIG
    field = _WORKER_FIELDS[agent]
    target = config["targets"][:, agent]
    params = make_local_params(
        state, neighbor_positions, previous_control, target,
        config["horizon"], config["Ts"], config["d_safe"],
        config["gamma_cbf"], config["rho_cbf"], config["robot_radius"],
        _WORKER_CORRIDORS, _WORKER_ROUNDABOUTS,
    )
    params["cost_function"] = wrapper_cost
    params["gradient_function"] = wrapper_grad

    nominal = make_field_nominal_plan(
        state, field, target, config["horizon"], config["Ts"],
        config["reference_speed"], config["omega_min"], config["omega_max"],
    )
    initial_cost = wrapper_cost(warm_start, params, field)
    nominal_cost = wrapper_cost(nominal, params, field)
    initial_guess = nominal if np.isfinite(nominal_cost) and nominal_cost < initial_cost else warm_start
    initial_cost = min(initial_cost, nominal_cost) if np.isfinite(nominal_cost) else initial_cost

    started = time.perf_counter()
    optimized, _, iterations = solver.solve(initial_guess, params, field)
    solver_seconds = time.perf_counter() - started
    cost, _, _, diagnostics = build_local_dcbf(optimized, params, field)
    if not np.isfinite(cost) or cost > initial_cost:
        optimized = initial_guess
        cost, _, _, diagnostics = build_local_dcbf(optimized, params, field)
    return (
        agent, optimized[:2], shift_control_sequence(optimized, config["horizon"]),
        solver, solver_seconds, iterations, diagnostics["max_dcbf_violation"],
        diagnostics["max_geofence_violation"], os.getpid(), control_started_ns,
        time.perf_counter_ns(),
    )


def _rectangle_vertices(x_min, x_max, y_min, y_max):
    return np.array([
        [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max],
    ])


def local_map_geometry():
    overlap = 0.1
    gap = 0.25
    bounds = [
        (-1.5, -0.8-gap, 0.3, 0.7),
        (-0.8+gap, 0.0+overlap, 0.3, 0.7),
        (0.0-overlap, 0.8-gap, 0.3, 0.7),
        (0.8+gap, 1.5, 0.3, 0.7),
        (-1.5, -0.8+overlap, -0.7, -0.3),
        (-0.8-overlap, 0.0-gap, -0.7, -0.3),
        (0.0+gap, 0.8+overlap, -0.7, -0.3),
        (0.8-overlap, 1.5, -0.7, -0.3),
        (-1.0, -0.6, -1.0, -0.5+overlap),
        (-1.0, -0.6, -0.5-overlap, 0.5-gap),
        (-1.0, -0.6, 0.5+gap, 1.0),
        (-0.2, 0.2, -1.0, -0.5-gap),
        (-0.2, 0.2, -0.5+gap, 0.5+overlap),
        (-0.2, 0.2, 0.5-overlap, 1.0),
        (0.6, 1.0, -1.0, -0.5+overlap),
        (0.6, 1.0, -0.5-overlap, 0.5-gap),
        (0.6, 1.0, 0.5+gap, 1.0),
    ]
    directions = [(1.0, 0.0)] * 8 + [(0.0, 1.0)] * 9
    corridors = [
        {"v": _rectangle_vertices(*item), "dir": np.asarray(directions[index])}
        for index, item in enumerate(bounds)
    ]
    roundabouts = np.array([
        [-0.8, 0.5, 0.35, 0.08, 1.0],
        [0.8, 0.5, 0.35, 0.08, 1.0],
        [0.0, -0.5, 0.35, 0.08, 1.0],
    ])
    return corridors, roundabouts


def scenario_config(agent_scenario):
    scenario = str(agent_scenario).lower()
    if scenario == "dois":
        starts = np.array([[-1.30, -0.80], [-0.50, -0.90]])
        targets = np.array([[1.30, -0.80], [-0.50, 0.90]])
        headings = np.array([0.0, np.pi / 2.0])
        return starts, targets, headings, 0.215, 2.0e7
    if scenario == "quatro":
        starts = np.array([[-1.30, -0.80, 0.00, -1.30],
                           [-0.50, -0.90, -0.90, 0.50]])
        targets = np.array([[1.30, -0.80, 0.00, 1.30],
                            [-0.50, 0.90, 0.90, 0.50]])
        headings = np.array([0.0, np.pi / 2.0, np.pi / 2.0, 0.0])
        return starts, targets, headings, 0.270, 2.0e8
    raise ValueError(f"Unknown agent scenario: {agent_scenario!r}")


def _load_or_generate_fields(cache_file, corridors, roundabouts, starts,
                             targets, bounds, resolution, robot_radius):
    cache_file = Path(cache_file)
    if cache_file.is_file():
        try:
            with np.load(cache_file, allow_pickle=True) as cache:
                valid = (
                    int(cache["cache_version"]) == 1
                    and np.array_equal(cache["starts"], starts)
                    and np.array_equal(cache["targets"], targets)
                    and np.array_equal(cache["bounds"], bounds)
                    and float(cache["resolution"]) == resolution
                    and float(cache["robot_radius"]) == robot_radius
                )
                if valid:
                    fields = list(cache["fields"])
                    route_names = list(cache["route_names"])
                    print("Vector fields loaded from cache.")
                    return fields, route_names
        except (OSError, KeyError, ValueError):
            pass

    fields = []
    route_names = []
    for agent in range(starts.shape[1]):
        print(f"Generating route and LBM field for agent {agent + 1}...")
        route_corridors, route_roundabouts, connections, names, *_ = planejar_rota(
            corridors, roundabouts, starts[:, agent], targets[:, agent]
        )
        X_grid, Y_grid, ux, uy = gerar_gvf_lbm(
            route_corridors, route_roundabouts, connections,
            targets[:, agent], bounds, resolution,
        )
        route_boxes = np.array([
            [item["v"][:, 0].min(), item["v"][:, 0].max(),
             item["v"][:, 1].min(), item["v"][:, 1].max()]
            for item in route_corridors
        ]).T
        geofence_h = build_route_geofence_grid(
            X_grid, Y_grid, route_boxes, route_roundabouts, robot_radius
        )
        fields.append({
            "X_grid": X_grid, "Y_grid": Y_grid,
            "x_vec": X_grid[0, :], "y_vec": Y_grid[:, 0],
            "ux": ux, "uy": uy, "route_boxes": route_boxes,
            "route_roundabouts": route_roundabouts, "geofence_h": geofence_h,
        })
        route_names.append(names)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file, cache_version=1, starts=starts, targets=targets,
        bounds=bounds, resolution=resolution, robot_radius=robot_radius,
        fields=np.asarray(fields, dtype=object),
        route_names=np.asarray(route_names, dtype=object),
    )
    print(f"Vector fields cached at {cache_file}.")
    return fields, route_names


def draw_map_on_robotarium(robotarium, corridors, roundabouts, starts, targets):
    ax = robotarium._axes_handle
    colors = plt.get_cmap("tab10")(np.arange(starts.shape[1]))
    angles = np.linspace(0.0, 2.0 * np.pi, 100)
    for corridor in corridors:
        ax.add_patch(PolygonPatch(
            corridor["v"], closed=True, facecolor=(0.82, 0.84, 0.87, 0.28),
            edgecolor=(0.35, 0.35, 0.35), linewidth=1.0, zorder=0,
        ))
        center = np.mean(corridor["v"], axis=0)
        dx, dy = corridor["dir"]
        ax.quiver(center[0] - 0.06 * dx, center[1] - 0.06 * dy,
                  0.12 * dx, 0.12 * dy, color="tab:blue",
                  angles="xy", scale_units="xy", scale=1, zorder=1)
    for cx, cy, outer_radius, inner_radius, direction in roundabouts:
        ax.fill(cx + outer_radius * np.cos(angles),
                cy + outer_radius * np.sin(angles),
                color=(0.82, 0.84, 0.87), alpha=0.28, zorder=0)
        ax.plot(cx + outer_radius * np.cos(angles),
                cy + outer_radius * np.sin(angles), color="0.35", zorder=1)
        ax.fill(cx + inner_radius * np.cos(angles),
                cy + inner_radius * np.sin(angles), color="white",
                edgecolor="black", zorder=1)
        for angle in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0):
            px = cx + 0.22 * np.cos(angle)
            py = cy + 0.22 * np.sin(angle)
            tx = -direction * np.sin(angle)
            ty = direction * np.cos(angle)
            ax.quiver(px - 0.045 * tx, py - 0.045 * ty,
                      0.09 * tx, 0.09 * ty, color="tab:blue",
                      angles="xy", scale_units="xy", scale=1, zorder=2)
    for agent in range(starts.shape[1]):
        ax.plot(*starts[:, agent], "o", color=colors[agent], markersize=6, zorder=4)
        ax.plot(*targets[:, agent], "p", color=colors[agent],
                markerfacecolor=colors[agent], markersize=9, zorder=4)
    ax.set_title(f"{starts.shape[1]} agents: individual LBM fields + DCBF")
    robotarium._fig.canvas.draw_idle()
    robotarium._fig.canvas.flush_events()


def _plot_results(results, starts, targets):
    X = results["X"]
    U = results["U"]
    distances = results["distance"]
    violations = results["dcbf_violation"]
    solver_times = results["solver_times"]
    Ts = results["Ts"]
    times = np.arange(X.shape[2]) * Ts
    colors = plt.get_cmap("tab10")(np.arange(X.shape[1]))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].grid(True)
    for agent in range(X.shape[1]):
        axes[0, 0].plot(X[0, agent], X[1, agent], color=colors[agent],
                        label=f"Agent {agent + 1}")
        axes[0, 0].plot(*starts[:, agent], "o", color=colors[agent])
        axes[0, 0].plot(*targets[:, agent], "p", color=colors[agent])
    axes[0, 0].set(title="Trajectories", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].legend()

    for pair_index, label in enumerate(results["pair_labels"]):
        axes[0, 1].plot(times, distances[pair_index], linewidth=1.2, label=label)
    axes[0, 1].axhline(results["d_safe"], color="tab:red", linestyle="--",
                       label="d_safe")
    axes[0, 1].grid(True)
    axes[0, 1].set(title="Inter-agent distances", xlabel="time [s]",
                   ylabel="distance [m]")
    axes[0, 1].legend(loc="best")

    axes[1, 0].semilogy(times, np.max(violations, axis=0) + np.finfo(float).eps)
    axes[1, 0].grid(True)
    axes[1, 0].set(title="Predicted DCBF violation", xlabel="time [s]",
                   ylabel="max(0, g_ij)")
    axes[1, 1].plot(times, 1e3 * solver_times.T, linewidth=1.0)
    axes[1, 1].axhline(1e3 * Ts, color="tab:red", linestyle="--", label="Ts")
    axes[1, 1].grid(True)
    axes[1, 1].set(title="PANOC time per agent", xlabel="time [s]",
                   ylabel="solver time [ms]")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    compute_figure, compute_axes = plt.subplots(
        3, 2, figsize=(14, 12), layout="constrained"
    )
    compute_time = times
    step_budget_ms = 1e3 * Ts
    solver_ms = 1e3 * solver_times
    total_solver_ms = np.sum(solver_ms, axis=0)
    batch_ms = 1e3 * results["controller_batch_times"]
    cycle_ms = 1e3 * results["cycle_wall_times"]
    pre_step_ms = 1e3 * results["pre_step_compute_times"]
    iteration_counts = results["solver_iterations"]
    active_solver_values = [solver_ms[index][solver_ms[index] > 0.0]
                            for index in range(solver_ms.shape[0])]
    p95 = np.array([
        np.percentile(values, 95) if values.size else np.nan
        for values in active_solver_values
    ])

    latency_axis = compute_axes[0, 0]
    for agent in range(solver_ms.shape[0]):
        latency_axis.plot(
            compute_time, solver_ms[agent], color=colors[agent], linewidth=1.0,
            label=f"Agent {agent + 1}",
        )
    latency_axis.axhline(step_budget_ms, color="tab:red", linestyle="--",
                         label="$T_s$ budget")
    latency_axis.set(title="PANOC latency by agent", xlabel="simulation time [s]",
                     ylabel="solver time [ms]")
    latency_axis.grid(True)
    latency_axis.legend(loc="upper right")

    dispatch_axis = compute_axes[0, 1]
    dispatch_axis.plot(compute_time, batch_ms, label="controller dispatch wall time")
    dispatch_axis.plot(compute_time, total_solver_ms, label="sum of worker solve times")
    dispatch_axis.axhline(step_budget_ms, color="tab:red", linestyle="--",
                          label="$T_s$ budget")
    dispatch_axis.set(title="Dispatch cost and aggregate solver work",
                      xlabel="simulation time [s]", ylabel="time [ms]")
    dispatch_axis.grid(True)
    dispatch_axis.legend(loc="upper right")

    cycle_axis = compute_axes[1, 0]
    cycle_axis.plot(compute_time, pre_step_ms, color="tab:purple",
                    label="computation before Robotarium.step")
    cycle_axis.plot(compute_time, cycle_ms, color="0.45", linewidth=0.9,
                    alpha=0.8, label="full cycle incl. simulator pacing")
    cycle_axis.axhline(step_budget_ms, color="tab:red", linestyle="--",
                       label="$T_s$ budget")
    over_budget = pre_step_ms > step_budget_ms
    cycle_axis.scatter(compute_time[over_budget], pre_step_ms[over_budget],
                       color="tab:red", s=14, label="budget overrun", zorder=3)
    cycle_axis.set(
        title=f"Compute budget overruns ({np.count_nonzero(over_budget)}/{pre_step_ms.size})",
        xlabel="simulation time [s]", ylabel="time [ms]",
    )
    cycle_axis.grid(True)
    cycle_axis.legend(loc="upper right")

    distribution_axis = compute_axes[1, 1]
    available = [(index, values) for index, values in enumerate(active_solver_values)
                 if values.size]
    if available:
        distribution_axis.boxplot(
            [values for _, values in available],
            tick_labels=[f"A{index + 1}" for index, _ in available],
            showmeans=True,
            showfliers=True,
        )
        distribution_axis.scatter(
            np.arange(1, len(available) + 1),
            p95[[index for index, _ in available]],
            marker="D", color="tab:red", label="p95", zorder=3,
        )
        distribution_axis.legend()
    else:
        distribution_axis.text(0.5, 0.5, "No solver samples",
                               ha="center", va="center",
                               transform=distribution_axis.transAxes)
    distribution_axis.set(title="Solver latency distribution",
                          xlabel="agent", ylabel="solver time [ms]")
    distribution_axis.grid(True, axis="y")

    iterations_axis = compute_axes[2, 0]
    for agent in range(iteration_counts.shape[0]):
        iterations_axis.step(
            compute_time, iteration_counts[agent], where="post",
            color=colors[agent], linewidth=1.0, label=f"Agent {agent + 1}",
        )
    iterations_axis.axhline(
        results["solver_max_iterations"], color="tab:red", linestyle="--",
        label="iteration limit",
    )
    iterations_axis.set(title="PANOC iterations per solve",
                        xlabel="simulation time [s]", ylabel="iterations")
    iterations_axis.grid(True)
    iterations_axis.legend(loc="upper right")

    overlap_axis = compute_axes[2, 1]
    overlap_ratio = np.divide(
        total_solver_ms, batch_ms,
        out=np.zeros_like(total_solver_ms), where=batch_ms > 0.0,
    )
    overlap_axis.plot(compute_time, overlap_ratio, color="tab:green",
                      label="sum solve time / dispatch wall time")
    overlap_axis.axhline(1.0, color="0.4", linestyle="--", label="ratio = 1")
    overlap_axis.axhline(
        solver_ms.shape[0], color="tab:blue", linestyle=":",
        label="agent count",
    )
    overlap_axis.set(title="Parallel overlap indicator",
                     xlabel="simulation time [s]", ylabel="time ratio")
    overlap_axis.grid(True)
    overlap_axis.legend(loc="upper right")
    compute_figure.suptitle(
        f"Computational performance | Numba warmup: "
        f"{1e3 * results['numba_warmup_seconds']:.1f} ms",
        fontsize=14,
    )
    return figure, compute_figure


def _save_results(results, output_path):
    saved_results = dict(results)
    saved_results["route_names"] = np.asarray(
        [" -> ".join(agent_route) for agent_route in results["route_names"]],
        dtype=str,
    )
    np.savez_compressed(output_path, **saved_results)


def main_multiagente_dcbf(
    show_figure=True, simulation_time=60.0, enable_dcbf=True,
    solver_max_iterations=5, agent_scenario="quatro", d_safe_override=None,
    rho_cbf_override=None, skip_initialization=True, parallel=True,
    cache_file=None, save_results=True,
):
    starts, targets, headings, scenario_d_safe, scenario_rho_cbf = scenario_config(agent_scenario)
    number_of_agents = starts.shape[1]
    initial_conditions = np.vstack((starts, headings))
    corridors, roundabouts = local_map_geometry()
    base_dir = Path(__file__).resolve().parent
    if cache_file is None:
        cache_file = base_dir / f"campos_{agent_scenario}_agentes.npz"
    fields, route_names = _load_or_generate_fields(
        cache_file, corridors, roundabouts, starts, targets,
        np.array([-1.6, 1.6, -1.0, 1.0]), 0.02, 0.05,
    )

    horizon = 30
    d_safe = scenario_d_safe if d_safe_override is None else float(d_safe_override)
    rho_cbf = (scenario_rho_cbf if rho_cbf_override is None else float(rho_cbf_override))
    rho_cbf *= bool(enable_dcbf)
    required_distance = 0.20
    gamma_cbf = 0.10
    reference_speed = 0.09
    goal_tolerance = 0.06
    v_min, v_max = 0.0, 0.10
    omega_min, omega_max = -1.5, 1.5
    robot_radius = 0.05
    time_step = Robotarium.TIME_STEP
    number_of_steps = int(np.ceil(simulation_time / time_step))

    solver_min = np.tile([v_min, omega_min], horizon)
    solver_max = np.tile([v_max, omega_max], horizon)
    solvers = [
        PanocSolver(0.01, 1e-4, solver_max_iterations, 1e-3,
                    solver_min, solver_max, 20)
        for _ in range(number_of_agents)
    ]
    warm_starts = [
        make_field_nominal_plan(
            initial_conditions[:, agent], fields[agent], targets[:, agent],
            horizon, time_step, reference_speed, omega_min, omega_max,
        )
        for agent in range(number_of_agents)
    ]
    warm_predictions = np.zeros((2, horizon + 1, number_of_agents))
    for agent in range(number_of_agents):
        warm_predictions[:, :, agent] = rollout_unicycle(
            initial_conditions[:, agent], warm_starts[agent], time_step, horizon
        )[:2]
    warm_neighbor_ids = np.flatnonzero(np.arange(number_of_agents) != 0)
    warm_params = make_local_params(
        initial_conditions[:, 0], warm_predictions[:, :, warm_neighbor_ids],
        np.zeros(2), targets[:, 0], horizon, time_step, d_safe, gamma_cbf,
        rho_cbf, robot_radius, corridors, roundabouts,
    )
    numba_warmup_started = time.perf_counter()
    wrapper_cost(warm_starts[0], warm_params, fields[0])
    wrapper_grad(warm_starts[0], warm_params, fields[0])
    numba_warmup_seconds = time.perf_counter() - numba_warmup_started
    print(
        "Numba DCBF/PANOC kernel compiled and warm-cached "
        f"in {1e3 * numba_warmup_seconds:.1f} ms."
    )

    previous_commands = np.zeros((2, number_of_agents))
    robotarium = Robotarium(
        number_of_robots=number_of_agents,
        show_figure=bool(show_figure),
        initial_conditions=initial_conditions,
        sim_in_real_time=True,
        skip_initialization=bool(skip_initialization),
    )
    if show_figure:
        draw_map_on_robotarium(robotarium, corridors, roundabouts, starts, targets)

    pair_indices = np.array([
        (i, j) for i in range(number_of_agents)
        for j in range(i + 1, number_of_agents)
    ], dtype=int).reshape(-1, 2)
    pair_labels = [f"A{i + 1}-A{j + 1}" for i, j in pair_indices]
    X_history = np.zeros((3, number_of_agents, number_of_steps))
    U_history = np.zeros((2, number_of_agents, number_of_steps))
    distance_history = np.full((len(pair_indices), number_of_steps), np.nan)
    dcbf_violation_history = np.zeros((number_of_agents, number_of_steps))
    geofence_violation_history = np.zeros((number_of_agents, number_of_steps))
    solver_time_history = np.zeros((number_of_agents, number_of_steps))
    solver_iterations = np.zeros((number_of_agents, number_of_steps), dtype=int)
    controller_batch_time_history = np.zeros(number_of_steps)
    pre_step_compute_time_history = np.zeros(number_of_steps)
    cycle_wall_time_history = np.zeros(number_of_steps)
    done = np.zeros(number_of_agents, dtype=bool)
    completion_times = np.full(number_of_agents, np.nan)
    unsafe_events = np.zeros(len(pair_indices), dtype=int)
    unsafe_warning_issued = False
    parallel_reported = False
    executed_steps = 0

    config = {
        "targets": targets, "horizon": horizon, "Ts": time_step,
        "d_safe": d_safe, "gamma_cbf": gamma_cbf, "rho_cbf": rho_cbf,
        "robot_radius": robot_radius, "reference_speed": reference_speed,
        "omega_min": omega_min, "omega_max": omega_max,
    }
    if parallel:
        executor = ProcessPoolExecutor(
            max_workers=number_of_agents,
            initializer=_initialize_controller_workers,
            initargs=(fields, corridors, roundabouts, config),
        )
    else:
        _initialize_controller_workers(fields, corridors, roundabouts, config)
        executor = None
    try:
        for step in range(number_of_steps):
            cycle_started = time.perf_counter()
            states = robotarium.get_poses()
            X_history[:, :, step] = states
            for agent in range(number_of_agents):
                if (not done[agent]
                        and np.linalg.norm(states[:2, agent] - targets[:, agent]) < goal_tolerance):
                    done[agent] = True
                    completion_times[agent] = step * time_step
                    warm_starts[agent] = np.zeros(2 * horizon)

            communicated = np.zeros((2, horizon + 1, number_of_agents))
            for agent in range(number_of_agents):
                plan = rollout_unicycle(
                    states[:, agent], warm_starts[agent], time_step, horizon
                )
                communicated[:, :, agent] = plan[:2]

            active_agents = np.flatnonzero(~done)
            jobs = []
            for agent in active_agents:
                neighbor_ids = np.flatnonzero(np.arange(number_of_agents) != agent)
                jobs.append((
                    int(agent), states[:, agent].copy(),
                    communicated[:, :, neighbor_ids].copy(),
                    previous_commands[:, agent].copy(), warm_starts[agent].copy(),
                    solvers[agent],
                ))
            commands = np.zeros((2, number_of_agents))
            new_plans = list(warm_starts)
            if jobs:
                controller_batch_started = time.perf_counter()
                if executor is None:
                    computed = [_compute_agent_control(job) for job in jobs]
                else:
                    computed = list(executor.map(
                        _compute_agent_control, jobs, chunksize=1
                    ))
                controller_batch_time_history[step] = (
                    time.perf_counter() - controller_batch_started
                )
                if not parallel_reported:
                    worker_pids = sorted({result[8] for result in computed})
                    overlapping = any(
                        max(first[9], second[9]) < min(first[10], second[10])
                        for index, first in enumerate(computed)
                        for second in computed[index + 1:]
                    )
                    print(f"Controller worker PIDs (first cycle): {worker_pids}")
                    print(f"Controller execution overlap observed: {overlapping}")
                    parallel_reported = True
                for (agent, command, plan, solver, solver_seconds, iterations,
                     dcbf_violation, geofence_violation, _, _, _) in computed:
                    commands[:, agent] = command
                    new_plans[agent] = plan
                    solvers[agent] = solver
                    solver_time_history[agent, step] = solver_seconds
                    solver_iterations[agent, step] = iterations
                    dcbf_violation_history[agent, step] = dcbf_violation
                    geofence_violation_history[agent, step] = geofence_violation

            commands[0] = np.clip(commands[0], v_min, v_max)
            commands[1] = np.clip(commands[1], omega_min, omega_max)
            robotarium.set_velocities(np.arange(number_of_agents), commands)
            pre_step_compute_time_history[step] = time.perf_counter() - cycle_started
            robotarium.step()
            U_history[:, :, step] = commands
            previous_commands = commands
            warm_starts = new_plans

            for pair_index, (first, second) in enumerate(pair_indices):
                distance = np.linalg.norm(states[:2, first] - states[:2, second])
                distance_history[pair_index, step] = distance
                if distance < required_distance:
                    unsafe_events[pair_index] += 1
                    if not unsafe_warning_issued:
                        print(
                            f"WARNING: required distance violated at t={step*time_step:.3f}s "
                            f"by {pair_labels[pair_index]} ({distance:.4f} m)."
                        )
                        unsafe_warning_issued = True
            cycle_wall_time_history[step] = time.perf_counter() - cycle_started
            executed_steps = step + 1
            if np.all(done):
                print(f"All {number_of_agents} agents reached their targets.")
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    X_history = X_history[:, :, :executed_steps]
    U_history = U_history[:, :, :executed_steps]
    distance_history = distance_history[:, :executed_steps]
    dcbf_violation_history = dcbf_violation_history[:, :executed_steps]
    geofence_violation_history = geofence_violation_history[:, :executed_steps]
    solver_time_history = solver_time_history[:, :executed_steps]
    solver_iterations = solver_iterations[:, :executed_steps]
    controller_batch_time_history = controller_batch_time_history[:executed_steps]
    pre_step_compute_time_history = pre_step_compute_time_history[:executed_steps]
    cycle_wall_time_history = cycle_wall_time_history[:executed_steps]
    if executed_steps == 0:
        raise RuntimeError("The simulation executed no time steps.")

    minimum_per_pair = np.nanmin(distance_history, axis=1)
    minimum_distance = float(np.nanmin(minimum_per_pair))
    measured_h = distance_history**2 - d_safe**2
    if executed_steps > 1:
        measured_g = (1.0 - gamma_cbf) * measured_h[:, :-1] - measured_h[:, 1:]
        measured_violation_per_pair = np.max(np.maximum(0.0, measured_g), axis=1)
        measured_violation = float(np.max(measured_violation_per_pair))
    else:
        measured_violation_per_pair = np.full(len(pair_indices), np.nan)
        measured_violation = float("nan")

    solved_times = solver_time_history[solver_time_history > 0.0]
    mean_panoc_time = float(np.mean(solved_times)) if solved_times.size else float("nan")
    total_solver_time_per_cycle = np.sum(solver_time_history, axis=0)
    final_errors = np.linalg.norm(X_history[:2, :, -1] - targets, axis=0)
    window_steps = min(executed_steps, int(np.ceil(3.0 / time_step)))
    mean_speed = np.mean(
        np.abs(U_history[0, :, -window_steps:]), axis=1
    )
    stalled = (~done) & (mean_speed < 0.005) & (final_errors > goal_tolerance)
    possible_deadlock = bool(np.any(stalled))

    print("\n=== RESULTS ===")
    print(f"Minimum measured distance       : {minimum_distance:.4f} m")
    print(f"Required physical distance      : {required_distance:.4f} m")
    print(f"Distance used by DCBF           : {d_safe:.4f} m")
    print(f"Maximum predicted DCBF violation: {np.max(dcbf_violation_history):.4e}")
    print(f"Maximum measured DCBF violation : {measured_violation:.4e}")
    print(f"Maximum geofence violation      : {np.max(geofence_violation_history):.4e} m")
    print(f"Mean PANOC time                 : {1e3*mean_panoc_time:.3f} ms")
    print(f"Mean sum of solver times/cycle  : {1e3*np.mean(total_solver_time_per_cycle):.3f} ms")
    print(f"Maximum PANOC iterations        : {np.max(solver_iterations)}")
    for agent, error in enumerate(final_errors):
        print(f"Final error agent {agent + 1:<2d}          : {error:.4f} m")
    print(f"Mission completed               : {done.tolist()}")
    print(f"Agents at target                : {int(np.sum(done))}/{number_of_agents}")
    print(f"Samples below required distance : {int(np.sum(unsafe_events))}")
    print(f"Possible safe deadlock          : {'YES' if possible_deadlock else 'NO'}")
    for label, distance in zip(pair_labels, minimum_per_pair):
        print(f"Minimum distance {label:<8}      : {distance:.4f} m")
    robotarium.debug()

    results = {
        "route_names": route_names, "starts": starts, "targets": targets,
        "Ts": time_step, "executed_steps": executed_steps,
        "simulated_time": executed_steps * time_step,
        "X": X_history, "U": U_history, "distance": distance_history,
        "pair_indices": pair_indices + 1, "pair_labels": pair_labels,
        "minimum_distance_per_pair": minimum_per_pair,
        "required_distance": required_distance, "d_safe": d_safe,
        "rho_cbf": rho_cbf, "goal_tolerance": goal_tolerance,
        "robot_radius": robot_radius,
        "max_dcbf_violation": float(np.max(dcbf_violation_history)),
        "max_measured_dcbf_violation": measured_violation,
        "max_measured_dcbf_violation_per_pair": measured_violation_per_pair,
        "max_geofence_violation": float(np.max(geofence_violation_history)),
        "minimum_distance": minimum_distance,
        "unsafe_distance_events_per_pair": unsafe_events,
        "total_unsafe_distance_events": int(np.sum(unsafe_events)),
        "dcbf_violation": dcbf_violation_history,
        "geofence_violation": geofence_violation_history,
        "solver_times": solver_time_history,
        "total_solver_time_per_cycle": total_solver_time_per_cycle,
        "controller_batch_times": controller_batch_time_history,
        "pre_step_compute_times": pre_step_compute_time_history,
        "cycle_wall_times": cycle_wall_time_history,
        "numba_warmup_seconds": numba_warmup_seconds,
        "mean_panoc_time": mean_panoc_time,
        "solver_iterations": solver_iterations,
        "solver_max_iterations": int(solver_max_iterations),
        "final_errors": final_errors, "completed": done,
        "completion_times": completion_times,
        "mean_speed_last_3s": mean_speed, "stalled_unfinished": stalled,
        "possible_safe_deadlock": possible_deadlock,
        "number_of_agents": number_of_agents,
        "number_completed": int(np.sum(done)), "scenario": str(agent_scenario),
        "dcbf_enabled": bool(enable_dcbf),
        "skipped_robotarium_initialization": bool(skip_initialization),
    }
    if save_results:
        output_path = base_dir / f"resultado_{agent_scenario}_agentes_python.npz"
        _save_results(results, output_path)
        print(f"Results saved to {output_path}.")
    if show_figure:
        _plot_results(results, starts, targets)
        plt.show(block=True)
    return results


def main(argv=None):
    freeze_support()
    parser = argparse.ArgumentParser(description="ECC2026 Robotarium Python simulation")
    parser.add_argument("--scenario", choices=("dois", "quatro"), default="quatro")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--run-initialization", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--cache-file", type=Path)
    args = parser.parse_args(argv)
    return main_multiagente_dcbf(
        agent_scenario=args.scenario,
        simulation_time=args.seconds,
        solver_max_iterations=args.solver_iterations,
        show_figure=not args.headless,
        parallel=not args.serial,
        skip_initialization=not args.run_initialization,
        save_results=not args.no_save,
        cache_file=args.cache_file,
    )


if __name__ == "__main__":
    main()
