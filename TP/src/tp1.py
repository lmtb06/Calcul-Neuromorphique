from itertools import cycle
from typing import Iterable, Sequence, List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation
from neuromorphic import LIF, NeuroneUpdateStrategy, EulerUpdateStrategy, RK4UpdateStrategy, SimulationNeurones, AnimationLine2DPlotter, SimulationEventType

def main() -> None:
    # Paramètres globaux
    U0: float = 0.0
    theta: float = 1.0e-1
    C: float = 1.0
    R: float = 1.0
    intensite_cour: float = 1.0e0
    N: int = 2
    duree: float = 1.0e-0
    base_dt: float = 1.0e-2

    # Création de la figure et des axes
    fig : Figure
    axes : np.ndarray
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10),sharex=True, sharey='col')
    fig.subplots_adjust(hspace=0.3)
    time_steps: List[float] = [base_dt / 10**i for i in range(3)]
    delta_t_simulations: List[np.ndarray] = [np.diff(np.arange(0, duree + dt, dt)) for dt in time_steps]

    for i, (axe, dt) in enumerate(zip(axes, time_steps)):
        axe_potentiel: Axes = axes[i,0]
        axe_potentiel.set_ylabel("Potentiel (V)", fontsize=10)
        axe_potentiel.set_xmargin(0.01)
        axe_potentiel.set_ymargin(0.01)
        # axe_potentiel.set_ylim(U0, theta)
        # axe_potentiel.set_xlim(0, duree)
        # axe_potentiel.autoscale(enable=True, axis='both', tight=True)

        axe_potentiel.grid(True, linestyle='solid', alpha=0.7)
        axe_potentiel.set_xlabel("Temps (s)", fontsize=10)
        axe_potentiel.set_title(f"Pas de temps simulation : {dt:.2e}s", fontsize=12)
        axe_potentiel.tick_params(axis='both', which='major', labelsize=8)
        axe_potentiel.relim()
        axe_potentiel.autoscale_view()

        axe_intensite: Axes = axes[i,1]
        axe_intensite.set_ylabel("Intensité (A)", fontsize=10)
        # axe_intensite.fmt_xdata = lambda x: f"{x:.2e-3}"
        # axe_intensite.fmt_ydata = lambda y: f"{y:.2e-3}"
        
        axe_intensite.set_xmargin(0.01)
        axe_intensite.set_xlim(0, duree)
        # axe_intensite.autoscale(enable=True, axis='both', tight=True)
        axe_intensite.grid(True, linestyle='solid', alpha=0.7)
        axe_intensite.set_xlabel("Temps (s)", fontsize=10)
        axe_intensite.set_title(f"Pas de temps simulation : {dt:.2e}s", fontsize=12)
        axe_intensite.relim()
        axe_intensite.autoscale_view()
        axe_intensite.tick_params(axis='both', which='major', labelsize=8)

    # Initialisation des neurones et des stratégies d'intégration
    update_cycle: Iterable[NeuroneUpdateStrategy] = cycle([EulerUpdateStrategy(), RK4UpdateStrategy()])
    neurones_par_simulation: List[List[LIF]] = [
        [LIF(U0=U0, theta=theta, C=C, R=R, U=0.99*theta) for _ in range(N)]
        for _ in range(len(time_steps))
    ]
    update_strategies_par_simulation: List[List[NeuroneUpdateStrategy]] = [
        [next(update_cycle) for _ in range(N)]
        for _ in range(len(time_steps))
    ]
    simulations: List[SimulationNeurones] = []
    plotters: List[AnimationLine2DPlotter] = []

    for i, (neurones, update_strategies) in enumerate(zip(neurones_par_simulation, update_strategies_par_simulation)):
        nom_neurones: List[str] = [f"N{j+1} - {update_strategies[j]} - dt {time_steps[i]:.1e}" for j in range(len(neurones))]
        simulation = SimulationNeurones(
            nom_neurone=nom_neurones,
            neurones=neurones,
            update_strategies=update_strategies
        )
        plotter_potentiel = AnimationLine2DPlotter(axes[i,0], nom_neurones=nom_neurones,champ="U")
        simulation.subscribe(event_type=SimulationEventType.INIT, subscriber=plotter_potentiel)
        simulation.subscribe(event_type=SimulationEventType.RUN_END, subscriber=plotter_potentiel)
        simulation.subscribe(event_type=SimulationEventType.UPDATE, subscriber=plotter_potentiel)
        plotters.append(plotter_potentiel)

        plotter_intensite = AnimationLine2DPlotter(axes[i,1], nom_neurones=nom_neurones,champ="I_ext")
        simulation.subscribe(event_type=SimulationEventType.INIT, subscriber=plotter_intensite)
        simulation.subscribe(event_type=SimulationEventType.RUN_END, subscriber=plotter_intensite)
        simulation.subscribe(event_type=SimulationEventType.UPDATE, subscriber=plotter_intensite)
        plotters.append(plotter_intensite)

        simulations.append(simulation)

    
    def init_func() -> Sequence[Artist]:
        # Réinitialisation de toutes les simulations
        artists: List[Artist] = []
        for sim in simulations:
            sim.reset()

        for plotter in plotters:
            artists.extend(plotter.last_updated_artists)
            plotter.clear_updated_artists()
        
        fig.canvas.flush_events()
        return artists

    def animate_func(frame: int) -> Sequence[Artist]:
        artists: List[Artist] = []
        for i, sim in enumerate(simulations):
            if frame >= len(delta_t_simulations[i]):
                continue
            delta_t_sim:float = delta_t_simulations[i][frame]

            sim.update(delta_t=delta_t_sim, current_inputs=[intensite_cour] * len(neurones_par_simulation[i]))

        for plotter in plotters:
            artists.extend(plotter.last_updated_artists)
            plotter.clear_updated_artists()
        
        fig.canvas.flush_events()
        return artists

    max_frames = max(len(ts) for ts in delta_t_simulations)
    ani = FuncAnimation(
        fig,
        animate_func,
        init_func=init_func,
        frames=max_frames,
        interval=0,
        blit=True,
        repeat=True,
    )

    fig.suptitle(
        "Simulation des neurones LIF avec différentes stratégies d'intégration et pas de temps",
    )
    # ani.save("simulation.gif", dpi=300, writer="imagemagick", fps=30)
    # fig.legend(
    #     loc="upper right",
    #     fontsize=8,
    #     # bbox_to_anchor=(1.1, 1),
    #     # borderaxespad=0,
    #     # frameon=False,
    # )
    plt.tight_layout()
    plt.show()

    # plt.savefig("simulation.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
