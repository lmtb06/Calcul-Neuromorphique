from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Optional
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter, FuncFormatter
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.units import AxisInfo, ConversionInterface, registry
from numpy.typing import ArrayLike
from .simulation import SimulationEventType, SimulationNeurones
from .etat_neurone import EtatNeurone
from .neurone import Neurone

class NeuronesPlotter(ABC):
    def __init__(self, axes: Axes, ) -> None:
        self._axes: Axes = axes
        self._axes.grid()
        self._nb_neurones: int = 0

    def _check_limits(self, new_x_limits: tuple[float, float], new_y_limits: tuple[float, float]) -> None:
        x_limits: tuple[float, float] = self._axes.get_xlim()
        y_limits: tuple[float, float] = self._axes.get_ylim()
        rescale : bool = False
        if new_x_limits[0] < x_limits[0] or new_x_limits[1] > x_limits[1]:
            self._axes.set_xlim(new_x_limits)
            rescale = True
        if new_y_limits[0] < y_limits[0] or new_y_limits[1] > y_limits[1]:
            self._axes.set_ylim(new_y_limits)
            rescale = True

        if rescale:
            self._axes.relim()
            self._axes.autoscale_view()
            self._axes.figure.canvas.draw_idle()

    @abstractmethod
    def init(self) -> Iterable[Artist]:...

    @abstractmethod
    def draw(self) -> Iterable[Artist]:...

class LinearDataPlotter(NeuronesPlotter):
    def __init__(self, axes: Axes, title:str, x_label:str, y_label:str, x_unit:str, y_unit:str, x_data_field:str, y_data_field:str) -> None:
        super().__init__(axes)
        self._axes.set_title(title)
        self._axes.set_xlabel(x_label)
        self._axes.set_ylabel(y_label)
        self._x_data_field: str = x_data_field
        self._y_data_field: str = y_data_field
        self._axes.xaxis.set_major_formatter(EngFormatter(x_unit))
        self._axes.yaxis.set_major_formatter(EngFormatter(y_unit))
        self._plot_lines: list[Line2D] = []
    

    def init(self) -> Iterable[Artist]:

        for line in self._plot_lines:
            line.set_xdata([])
            line.set_ydata([])

        if self._nb_neurones > len(self._plot_lines):
            for _ in range(self._nb_neurones - len(self._plot_lines)):
                line: Line2D = self._axes.plot([], [])[0]
                line.set_animated(True)
                self._plot_lines.append(line)


        self._axes.legend(handles=self._plot_lines, labels=[f"Neurone {i+1}" for i in range(self._nb_neurones)])

        self._axes.relim()
        self._axes.autoscale_view()

        return *self._plot_lines,
    
    def draw(self) -> Iterable[Artist]:
        return *self._plot_lines,

    def update(self, event_type: SimulationEventType, context: SimulationNeurones, data) -> None:
        """
        Update the plot based on the event type and data received.
        """
        match event_type:
            case SimulationEventType.INIT:
                self._nb_neurones = len(data)
            case SimulationEventType.UPDATE:
                plot_data: np.ndarray = np.array([
                    serie_etats_neurones.etats[:context.iteration+1] for serie_etats_neurones in context.donnees
                ])
                timestamps: np.ndarray = plot_data[self._x_data_field]
                potentiels: np.ndarray = plot_data[self._y_data_field]
                for i, line in enumerate(self._plot_lines):
                    line.set_xdata(timestamps[i])
                    line.set_ydata(potentiels[i])
                
                new_x_limits: tuple[float, float] = (np.min(timestamps), np.max(timestamps))
                new_y_limits: tuple[float, float] = (np.min(potentiels), np.max(potentiels))
                self._check_limits(new_x_limits, new_y_limits)
    
class PotentielsPlotter(LinearDataPlotter):
    def __init__(self, axes: Axes) -> None:
        super().__init__(axes, "Potentiels des neurones", "temps", "potentiel", "s", "V", "t", "U")
    
class InputsPlotter(LinearDataPlotter):
    def __init__(self, axes: Axes) -> None:
        super().__init__(axes, "Courants d'entrée", "temps", "courant", "s", "A", "t", "I_ext")