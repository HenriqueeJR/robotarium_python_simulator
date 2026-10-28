from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .PanocSolver import PanocSolver
from .main_multiagente_dcbf import (
    _compute_agent_control,
    _initialize_controller_workers,
)


def test_process_pool_runs_one_local_controller_per_agent():
    agent_count = 4
    horizon = 4
    x_vec = np.linspace(-1.0, 1.0, 31)
    y_vec = np.linspace(-1.0, 1.0, 31)
    X, Y = np.meshgrid(x_vec, y_vec)
    fields = [{
        "x_vec": x_vec, "y_vec": y_vec,
        "ux": np.full_like(X, 0.04), "uy": np.zeros_like(Y),
    } for _ in range(agent_count)]
    corridors = [{"v": np.array([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
    ]), "dir": np.array([1.0, 0.0])}]
    roundabouts = np.empty((0, 5))
    targets = np.array([
        [0.7, -0.7, 0.7, -0.7],
        [0.7, 0.7, -0.7, -0.7],
    ])
    config = {
        "targets": targets, "horizon": horizon, "Ts": 0.033,
        "d_safe": 0.27, "gamma_cbf": 0.1, "rho_cbf": 2.0e8,
        "robot_radius": 0.05, "reference_speed": 0.09,
        "omega_min": -1.5, "omega_max": 1.5,
    }
    lower = np.tile([0.0, -1.5], horizon)
    upper = np.tile([0.1, 1.5], horizon)
    states = np.array([
        [-0.5, 0.5, -0.5, 0.5],
        [-0.5, -0.5, 0.5, 0.5],
        [0.0, np.pi, 0.0, np.pi],
    ])
    warm_start = np.tile([0.04, 0.0], horizon)
    jobs = []
    for agent in range(agent_count):
        neighbors = np.repeat(
            states[:2, np.arange(agent_count) != agent][:, None, :],
            horizon + 1,
            axis=1,
        )
        solver = PanocSolver(0.01, 1e-4, 2, 1e-3, lower, upper, 5)
        jobs.append((
            agent, states[:, agent], neighbors, np.zeros(2), warm_start, solver,
        ))

    with ProcessPoolExecutor(
        max_workers=agent_count,
        initializer=_initialize_controller_workers,
        initargs=(fields, corridors, roundabouts, config),
    ) as pool:
        outputs = list(pool.map(_compute_agent_control, jobs, chunksize=1))

    assert len(outputs) == agent_count
    assert {output[0] for output in outputs} == set(range(agent_count))
    for output in outputs:
        (_, command, plan, _, solver_time, iterations, dcbf, geofence,
         worker_pid, started_ns, finished_ns) = output
        assert command.shape == (2,)
        assert plan.shape == (2 * horizon,)
        assert np.all(np.isfinite(command))
        assert np.all(np.isfinite(plan))
        assert solver_time >= 0.0
        assert iterations >= 1
        assert dcbf >= 0.0
        assert geofence >= 0.0
        assert worker_pid > 0
        assert finished_ns >= started_ns
