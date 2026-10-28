# Four-agent baseline

This scenario reproduces the validated MATLAB four-agent baseline without introducing priority, flow-economy, coalitions, ADMM, centralized optimization, or waypoint tracking. The four robots start on two horizontal and two vertical lanes and consider the other three robots as neighbors.

```text
Agent 1: (-1.30,-0.50) -> ( 1.30,-0.50)
Agent 2: (-0.80,-0.90) -> (-0.80, 0.90)
Agent 3: ( 0.00,-0.90) -> ( 0.00, 0.90)
Agent 4: (-1.30, 0.50) -> ( 1.30, 0.50)
```

The Python port retains the MATLAB values: `Ts=0.033 s`, horizon 30, five PANOC iterations by default, physical separation 0.200 m, DCBF distance 0.270 m, and DCBF weight `2.0e8`.

Run from the repository root after installing the local requirements:

```powershell
.\.venv312\Scripts\python.exe -m rps.examples.ECC2026_python.executar_simulacao_quatro_agentes
```

The simulation skips the preliminary Robotarium initialization and begins at the defined experiment poses, matching the MATLAB local experiment. Use `skip_initialization=False` when calling `main_quatro_agentes_dcbf` to run that phase.

The MATLAB baseline's validated metrics are documented in `ECC2026_matlab/README_QUATRO_AGENTES.md`. They are not asserted as Python outcomes: the LBM field is regenerated in NumPy/Shapely, so the Python run must independently meet arrival, minimum-distance, DCBF, and geofence criteria before being treated as equivalent.
