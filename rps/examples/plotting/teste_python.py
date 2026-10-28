
import argparse
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor
from rps.robotarium import Robotarium


def controller_local(data):
    started_ns = time.perf_counter_ns()
    # Dados do robô i
    pose_i, goal_i, neighbors_i = data

    x, y, theta = pose_i
    error = goal_i - np.array([x, y])

    # Exemplo simples de controlador proporcional
    v_ref = 1.0 * np.linalg.norm(error)
    theta_ref = np.arctan2(error[1], error[0])
    w_ref = 2.0 * np.arctan2(
        np.sin(theta_ref - theta),
        np.cos(theta_ref - theta)
    )

    # Saturação ilustrativa
    v_ref = np.clip(v_ref, 0.0, 0.1)
    w_ref = np.clip(w_ref, -2.0, 2.0)

    return np.array([v_ref, w_ref]), os.getpid(), started_ns, time.perf_counter_ns()


def intervals_overlap(results):
    for index, result in enumerate(results):
        for other in results[index + 1:]:
            if max(result[2], other[2]) < min(result[3], other[3]):
                return True
    return False


def controller_batch(data):
    pose, goal, neighbors, repetitions = data
    started_ns = time.perf_counter_ns()
    command = None
    for _ in range(repetitions):
        command = controller_local((pose, goal, neighbors))[0]
    return command, os.getpid(), started_ns, time.perf_counter_ns()


def validate_parallel_execution(pool, jobs, repeats=10, controls_per_job=250):
    single_results = list(pool.map(controller_local, jobs))
    single_pids = sorted({result[1] for result in single_results})
    print(f"Single-control batch PIDs: {single_pids}")
    print(f"Single-control overlap observed: {intervals_overlap(single_results)}")

    benchmark_jobs = [(*job, controls_per_job) for job in jobs]
    serial_started = time.perf_counter()
    for _ in range(repeats):
        for job in benchmark_jobs:
            controller_batch(job)
    serial_seconds = time.perf_counter() - serial_started

    parallel_started = time.perf_counter()
    worker_pids = set()
    observed_overlap = False
    for _ in range(repeats):
        results = list(pool.map(controller_batch, benchmark_jobs))
        worker_pids.update(result[1] for result in results)
        observed_overlap = observed_overlap or intervals_overlap(results)
    parallel_seconds = time.perf_counter() - parallel_started
    speedup = serial_seconds / parallel_seconds if parallel_seconds else float("inf")

    print(f"Repeated-control worker PIDs: {sorted(worker_pids)}")
    print(f"Repeated-control overlap observed: {observed_overlap}")
    print(
        f"Benchmark ({repeats * len(jobs) * controls_per_job} control calculations): "
        f"serial={serial_seconds:.4f}s, process pool={parallel_seconds:.4f}s, "
        f"speedup={speedup:.2f}x"
    )


def validate_parallel_only():
    poses = np.array([
        [-0.5, 0.5, -0.5, 0.5],
        [-0.3, -0.3, 0.3, 0.3],
        [0.0, np.pi / 2, np.pi, -np.pi / 2],
    ])
    goals = np.array([
        [-0.4, -0.2],
        [0.4, -0.2],
        [-0.4, 0.2],
        [0.4, 0.2],
    ]).T
    jobs = [(poses[:, i], goals[:, i], []) for i in range(poses.shape[1])]

    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        validate_parallel_execution(pool, jobs)


def main():
    N = 4
    r = Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

    goals = np.array([
        [-0.4, -0.2],
        [ 0.4, -0.2],
        [-0.4,  0.2],
        [ 0.4,  0.2]
    ]).T
    r._axes_handle.scatter(goals[0], goals[1], marker="x", color="tab:red", zorder=5)
    r._axes_handle.set_title("Robotarium: real-time parallel control")
    r._fig.canvas.draw_idle()

    with ProcessPoolExecutor(max_workers=N) as pool:
        validated = False
        for _ in range(1000):
            poses = r.get_poses()

            # Exemplo: cada controlador recebe o estado local.
            # Substituir pela seleção real de vizinhos.
            jobs = [
                (
                    poses[:, i].copy(),
                    goals[:, i].copy(),
                    []
                )
                for i in range(N)
            ]

            results = list(
                pool.map(controller_local, jobs)
            )
            if not validated:
                validate_parallel_execution(pool, jobs)
                validated = True

            # A API recebe velocidades no formato 2 x N.
            # Verifique a convenção de comandos e as transformações
            # exigidas pela versão do simulador utilizada.
            velocities = np.column_stack([result[0] for result in results])
            r.set_velocities(np.arange(N), velocities)
            r.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-parallel",
        action="store_true",
        help="benchmark the control function without starting the simulator",
    )
    args = parser.parse_args()
    if args.validate_parallel:
        validate_parallel_only()
    else:
        main()
