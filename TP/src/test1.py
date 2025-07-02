from itertools import cycle
from typing import Iterable, Sequence
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from neuromorphic import LIF, RK4UpdateStrategy, EulerUpdateStrategy, SimulationNeurones, NeuronesPlotter, SimulationEventType, PotentielsPlotter, InputsPlotter

# Paramètres globaux
U0: float = 0.0
theta: float = 1.0e-1
C: float = 1.0
R: float = 1.0
intensite_cour: float = 1.0e0
N: int = 2
duree: float = 1.0e-0
base_dt: float = 1.0e-2

# plt.ion()
fig : Figure = plt.figure(layout='constrained')
axes = fig.subplots(2, 1)
axes0: Axes = axes[0]
axes1: Axes = axes[1]

# neurones_plotter: NeuronesPlotter = NeuronesPlotter(fig)
potentiels_plotter: PotentielsPlotter = PotentielsPlotter(axes0)
inputs_plotter: InputsPlotter = InputsPlotter(axes1)

cycler = cycle([EulerUpdateStrategy, RK4UpdateStrategy])
simulation: SimulationNeurones = SimulationNeurones(
    neurones=[LIF(U0=U0, theta=theta, C=C, R=R, I_ext=intensite_cour) for _ in range(N)],
    update_strategies=[next(cycler) for _ in range(N)],
)

for event_type in SimulationEventType:
    simulation.subscribe(event_type, potentiels_plotter)
    simulation.subscribe(event_type, inputs_plotter)

simulation.init(nb_iterations=int(duree/base_dt), delta_t=base_dt, get_current_inputs_callback=lambda t: [intensite_cour]*N)

def init_animation() -> Iterable[Artist]:
    simulation.reset()
    updated_artists: Iterable[Artist] = *(potentiels_plotter.init()), *(inputs_plotter.init())
    fig.canvas.flush_events()
    return updated_artists
    

def animate(frame: int) -> Iterable[Artist]:
    simulation.update()
    updated_artists: Iterable[Artist] = *(potentiels_plotter.draw()), *(inputs_plotter.draw())
    fig.canvas.flush_events()

    return updated_artists

ani = animation.FuncAnimation(
    fig,
    animate,
    init_func=init_animation,
    frames=int(duree/base_dt),
    interval=0,
    blit=True,
    repeat=True,
)

plt.show()