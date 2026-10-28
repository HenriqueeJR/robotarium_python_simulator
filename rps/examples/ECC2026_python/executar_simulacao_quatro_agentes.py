from __future__ import annotations

from multiprocessing import freeze_support

from .main_quatro_agentes_dcbf import main_quatro_agentes_dcbf


def main():
    freeze_support()
    results = main_quatro_agentes_dcbf(show_figure=True, simulation_time=100.0)
    print(
        "\nSimulation finished: "
        f"{results['number_completed']} of {results['number_of_agents']} "
        f"agents completed."
    )


if __name__ == "__main__":
    main()
