# MATLAB Codegen and Python validation

The MATLAB sources include a 350-element packed parameter vector, MATLAB Coder-compatible cost and gradient functions, platform-specific MEX binaries, and tests comparing the interpreted MATLAB and MEX implementations.

Python does not load MATLAB MEX binaries or use MATLAB Coder. This port retains `pack_params_1D.py` and its neighbor round-trip test to preserve the established parameter layout, while the actual Python controller evaluates the same cost and analytic gradient directly through `build_local_dcbf.py`.

Use these checks instead of generating MEX:

```powershell
.\.venv312\Scripts\python.exe -m pytest rps/examples/ECC2026_python/test_equivalencia_codegen.py
.\.venv312\Scripts\python.exe -m pytest rps/examples/ECC2026_python/test_gradiente_dcbf.py
```

`gerar_mex_local.py` is retained as an explicit compatibility entry point and raises an explanatory error; it does not claim to compile or validate a MEX binary.
