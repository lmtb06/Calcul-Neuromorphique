from abc import ABC, abstractmethod
import copy
from typing import Callable, Protocol, Sequence
from typing_extensions import override

from .neurone import Neurone
from .etat_neurone import SerieEtatsNeurone
from .neurone_update_strategy import NeuroneUpdateStrategy
from enum import Enum


class SimulationEventType(Enum):
    NEURONE_SPIKE = "neurone_spike"
    INIT = "init"
    RUN_START = "start"
    UPDATE = "update"
    RUN_END = "end"
    RESET = "reset"


class Subscriber(Protocol):
    def update(self, event_type: SimulationEventType, context, data) -> None:
        ...

class Publisher(ABC):
    def __init__(self):
        self._subscribers: dict[SimulationEventType, set[Subscriber]] = {
            event_type: set() for event_type in SimulationEventType
        }

    def subscribe(self, event_type: SimulationEventType, subscriber: Subscriber) -> None:
        """
        Add an subscriber to the subject.

        :param subscriber: The subscriber to add.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(subscriber)

    def unsubscribe(self, event_type: SimulationEventType, subscriber: Subscriber) -> None:
        """
        Remove an subscriber from the subject.

        :param subscriber: The subscriber to remove.
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(subscriber)

    def notify(self, event_type: SimulationEventType, data=None) -> None:
        """
        Notify all subscribers with the given data.

        :param data: The data to notify subscribers with.
        """
        for subscriber in self._subscribers.get(event_type, []):
            subscriber.update(event_type, self, data)

class Simulation(Publisher):
    @property
    @abstractmethod
    def neurones(self) -> list[Neurone]:
        ...

    @property
    @abstractmethod
    def donnees(self) -> list[SerieEtatsNeurone]:
        ...

    @property
    @abstractmethod
    def iteration(self) -> int:
        ...

    @abstractmethod
    def init(self, nb_iterations:int, delta_t:float, get_current_inputs_callback: Callable[[float], list[float]]) -> None:...

    @abstractmethod
    def run(self) -> None:...

    @abstractmethod
    def update(self) -> None:...

    @abstractmethod
    def reset(self) -> None:...

class SimulationNeurones(Simulation):
    def __init__(
        self,
        neurones: Sequence[Neurone],
        update_strategies: Sequence[NeuroneUpdateStrategy]
    ) -> None:
        super().__init__()
        self._neurones_initiaux: list[Neurone] = [copy.copy(neurone) for neurone in neurones]
        self._update_strategies: list[NeuroneUpdateStrategy] = [
            copy.copy(update_strategy) for update_strategy in update_strategies
        ]

        self._neurones_run: list[Neurone] = []
        self._donnees_neurones: list[SerieEtatsNeurone] = []
        self._get_current_inputs: Callable[[float], list[float]]
        self._delta_t: float
        self._nb_iterations: int
        self._iteration: int

    def _set_initial_values(self, nb_iterations:int, delta_t:float, get_current_inputs: Callable[[float], list[float]]) -> None:
        self._neurones_run = copy.deepcopy(self._neurones_initiaux)
        self._donnees_neurones = [
            SerieEtatsNeurone(steps=nb_iterations, type_elems=float)
            for _ in range(len(self._neurones_run))
        ]
        self._get_current_inputs = get_current_inputs
        self._delta_t = delta_t
        self._nb_iterations = nb_iterations
        self._iteration = 0

    @property
    def neurones(self) -> list[Neurone]:
        return self._neurones_run

    @property
    def donnees(self) -> list[SerieEtatsNeurone]:
        return self._donnees_neurones
    
    @property
    def iteration(self) -> int:
        return self._iteration

    @override
    def init(self, nb_iterations:int, delta_t:float, get_current_inputs_callback: Callable[[float], list[float]]) -> None:
        self._set_initial_values(nb_iterations, delta_t, get_current_inputs_callback)
        self.notify(SimulationEventType.INIT, self._donnees_neurones)

    @override
    def run(self) -> None:

        self.notify(SimulationEventType.RUN_START)
        for i in range(self._nb_iterations):
            self.update()
        
        self.notify(SimulationEventType.RUN_END)

    @override
    def update(self) -> None:
        if self._iteration >= self._nb_iterations:
            raise RuntimeError("Simulation has already ended.")
        
        t = self._iteration * self._delta_t
        current_inputs = self._get_current_inputs(t)

        for i, neurone in enumerate(self._neurones_run):
            # Update neuron state and check if it spikes
            spiked = self._update_strategies[i].update(
                neurone, self._delta_t, current_inputs[i]
            )

            self._donnees_neurones[i].set( 
                self._iteration, (neurone.etat, t)
            )

            if spiked:
                self.notify(SimulationEventType.NEURONE_SPIKE, neurone.etat)

        self.notify(SimulationEventType.UPDATE)

        self._iteration += 1

    @override
    def reset(self) -> None:
        """
        Reset the simulation.
        """
        self._set_initial_values(self._nb_iterations, self._delta_t, self._get_current_inputs)

        self.notify(SimulationEventType.RESET)
