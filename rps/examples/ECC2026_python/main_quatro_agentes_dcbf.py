from __future__ import annotations

from .main_multiagente_dcbf import main_multiagente_dcbf


def main_quatro_agentes_dcbf(
    show_figure=True,
    simulation_time=60.0,
    solver_max_iterations=5,
    d_safe=0.27,
    rho_cbf=2.0e8,
    skip_initialization=True,
    parallel=True,
    **kwargs,
):
    return main_multiagente_dcbf(
        show_figure=show_figure,
        simulation_time=simulation_time,
        enable_dcbf=True,
        solver_max_iterations=solver_max_iterations,
        agent_scenario="quatro",
        d_safe_override=d_safe,
        rho_cbf_override=rho_cbf,
        skip_initialization=skip_initialization,
        parallel=parallel,
        **kwargs,
    )
