from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .main_multiagente_dcbf import _plot_results, _save_results


def test_final_performance_dashboard_renders_headlessly():
    plt.switch_backend("Agg")
    agents = 4
    steps = 3
    results = {
        "X": np.zeros((3, agents, steps)),
        "U": np.zeros((2, agents, steps)),
        "distance": np.full((6, steps), 0.6),
        "dcbf_violation": np.zeros((agents, steps)),
        "solver_times": np.full((agents, steps), 0.004),
        "solver_iterations": np.full((agents, steps), 3),
        "solver_max_iterations": 5,
        "controller_batch_times": np.array([0.010, 0.012, 0.011]),
        "pre_step_compute_times": np.array([0.020, 0.035, 0.025]),
        "cycle_wall_times": np.array([0.030, 0.035, 0.032]),
        "numba_warmup_seconds": 0.2,
        "Ts": 0.033,
        "d_safe": 0.27,
        "pair_labels": [f"A{i + 1}-A{j + 1}"
                        for i in range(agents) for j in range(i + 1, agents)],
    }
    starts = np.zeros((2, agents))
    targets = np.ones((2, agents))

    trajectory_figure, performance_figure = _plot_results(results, starts, targets)
    try:
        assert len(trajectory_figure.axes) == 4
        assert len(performance_figure.axes) == 6
        titles = {axis.get_title() for axis in performance_figure.axes}
        assert "PANOC latency by agent" in titles
        assert "Dispatch cost and aggregate solver work" in titles
        assert "Compute budget overruns (1/3)" in titles
        assert "Solver latency distribution" in titles
        assert "PANOC iterations per solve" in titles
        assert "Parallel overlap indicator" in titles
    finally:
        plt.close(trajectory_figure)
        plt.close(performance_figure)


def test_results_save_with_different_route_lengths(tmp_path):
    output_path = tmp_path / "results.npz"
    results = {
        "route_names": [
            ["C5", "C6", "R3", "C7", "C8"],
            ["C9", "C10", "R1", "C11"],
            ["C12", "R3", "C13", "C14"],
            ["C1", "R1", "C2", "C3", "R2", "C4"],
        ],
        "completed": np.array([True, True, True, True]),
    }

    _save_results(results, output_path)

    with np.load(output_path, allow_pickle=False) as saved:
        assert saved["route_names"].shape == (4,)
        assert saved["route_names"][1] == "C9 -> C10 -> R1 -> C11"
        assert np.array_equal(saved["completed"], results["completed"])
