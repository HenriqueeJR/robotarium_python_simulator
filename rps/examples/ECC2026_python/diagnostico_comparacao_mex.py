from __future__ import annotations

import numpy as np

from .build_local_dcbf import build_local_dcbf


def diagnostico_comparacao_mex(U, params, field, finite_diff_step=1e-6,
                               gradient_samples=None):
    """Python replacement for MATLAB-vs-MEX and finite-difference diagnostics."""
    U = np.asarray(U, dtype=float).reshape(-1)
    cost, gradient, prediction, diagnostics = build_local_dcbf(U, params, field)
    indices = np.arange(U.size)
    if gradient_samples is not None and gradient_samples < U.size:
        indices = np.unique(np.linspace(
            0, U.size - 1, int(gradient_samples)
        ).round().astype(int))
    finite_difference = np.full(U.size, np.nan)
    for index in indices:
        direction = np.zeros_like(U)
        direction[index] = finite_diff_step
        plus = build_local_dcbf(U + direction, params, field, False)
        minus = build_local_dcbf(U - direction, params, field, False)
        finite_difference[index] = (plus - minus) / (2.0 * finite_diff_step)
    selected_gradient = gradient[indices]
    selected_fd = finite_difference[indices]
    error = np.linalg.norm(selected_gradient - selected_fd, ord=np.inf)
    relative_error = error / max(
        1.0, np.max(np.abs(selected_gradient)), np.max(np.abs(selected_fd))
    )
    return {
        "cost": cost,
        "gradient": gradient,
        "finite_difference_gradient": finite_difference,
        "gradient_abs_inf_error": error,
        "gradient_relative_error": relative_error,
        "predicted_states": prediction,
        "diagnostics": diagnostics,
        "passed": bool(relative_error < 5e-3),
    }
