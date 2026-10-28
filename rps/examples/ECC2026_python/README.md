# ECC2026 Python port

This package ports every MATLAB source module in `ECC2026_matlab` to Python. It retains the MATLAB four-agent baseline: a unicycle model, directed corridor/roundabout route graph, D2Q9 LBM vector fields, local NMPC/PANOC, analytic geofence and roundabout penalties, neighbor-prediction DCBF, warm starts, safety measurements, plots, and saved results. It does not add coalition optimization, priority rules, or ADMM because those are not part of the MATLAB baseline described by its README.

## Setup

Install the Python port's additional geometry dependency into the project environment:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r rps/examples/ECC2026_python/requirements.txt
```

## Run

From the repository root:

```powershell
.\.venv312\Scripts\python.exe -m rps.examples.ECC2026_python.executar_simulacao_quatro_agentes
```

The simulator opens its live Robotarium plot, starts at the four experiment poses, and runs the local controllers in a persistent Windows-compatible process pool. The Numba objective and gradient kernels are compiled and warmed once in the parent process before workers start; Numba caches the native kernels for later runs. It caches the regenerated fields in `campos_quatro_agentes.npz` and writes `resultado_quatro_agentes_python.npz` next to these sources.

For a quick headless end-to-end check, run one time step without saving outputs:

```powershell
.\.venv312\Scripts\python.exe -m rps.examples.ECC2026_python.main_multiagente_dcbf --seconds 0.033 --solver-iterations 1 --headless --no-save --cache-file "$env:TEMP\ecc2026_fields.npz"
```

The first simulation cycle prints worker process IDs and whether the four local controller executions had overlapping wall-clock intervals. `--serial` runs the same solves in the main process for comparison.

The generic entry point also supports the two-agent scenario and serial execution:

```python
from rps.examples.ECC2026_python.main_multiagente_dcbf import main_multiagente_dcbf

results = main_multiagente_dcbf(
    agent_scenario="quatro",
    simulation_time=60,
    solver_max_iterations=5,
    parallel=True,
)
```

For a short headless run, pass `show_figure=False`, `simulation_time=1.0`, and `save_results=False`.

## Validate

```powershell
.\.venv312\Scripts\python.exe -m pip install -r rps/examples/ECC2026_python/requirements-dev.txt
.\.venv312\Scripts\python.exe -m pytest rps/examples/ECC2026_python
```

The tests verify parameter packing, objective wrappers, equivalence between the Numba kernel and the retained Python reference, the analytic gradient against finite differences, and parallel controller workers. `test_equivalencia_codegen.py` replaces MATLAB source-versus-MEX checks with Python objective/gradient and packing checks; MATLAB Coder MEX binaries are not loaded by Python.

A warmed microbenchmark on the bundled horizon-8 test case measured 100 full cost-and-gradient evaluations at 0.0012 s with Numba versus 0.0605 s with the Python reference (51.8x for that kernel case). First-run JIT compilation is excluded; full simulation speedup depends on process-pool and Robotarium overhead.

A one-step, one-PANOC-iteration Robotarium smoke test with the default horizon 30 measured 1.35 ms per controller in parallel and 1.29 ms in serial; both modes produced identical poses, distances, and safety diagnostics for that step. This is a kernel/integration check, not a full-mission timing result.

At experiment end, the computational dashboard plots per-agent solve latency, dispatch wall time against aggregate worker time, pre-step compute versus the `Ts` budget, full cycle time including Robotarium pacing, per-agent latency distributions with p95 markers, PANOC iteration counts, and a parallel-overlap ratio. The compute-overrun count excludes the simulator's intentional real-time wait inside `step()`.

## Parallel execution

Only the main simulation's per-agent controller solve is dispatched through `ProcessPoolExecutor`. Each worker owns a persistent PANOC solver. At every simulation step, the main process freezes each agent's warm-start prediction for its neighbors, sends one local problem per unfinished agent, and collects commands before calling `Robotarium.set_velocities()` and `step()`. Pass `parallel=False` to run the same controller solves serially for comparison.

The MATLAB four-agent result in `ECC2026_matlab/README_QUATRO_AGENTES.md` is a reference, not a Python validation result. Compare Python arrival times, minimum pair distances, DCBF/geofence violations, and controller timing before claiming numerical parity.
