from __future__ import annotations


def gerar_mex_local():
    """There is no MATLAB Coder MEX stage in the Python implementation."""
    raise RuntimeError(
        "Python runs the validated objective and gradient directly. "
        "Use pytest on test_equivalencia_codegen.py and test_gradiente_dcbf.py "
        "instead of generating MATLAB Coder MEX files."
    )
