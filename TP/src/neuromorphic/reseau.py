import numpy as np
from .neurone import Neurone, LIF
from .neurone_update_strategy import NeuroneUpdateStrategy, EulerUpdateStrategy
from typing import List, Dict, Callable, Optional

def dirac(t: float) -> float:
    return 1. if t == 0. else 0.

class Reseau:
    def __init__(self, 
                 neurones: List[Neurone] = [LIF()], 
                 connectivite: Dict[int, Dict[int, float]] = {}, 
                 fonction_alpha: Optional[Callable[[float], float]] = None,
                 update_strategy: Optional[NeuroneUpdateStrategy]=None) -> None:
        self.connectivite: Dict[int, List[int]] = {i: list(connexions.keys()) for i, connexions in connectivite.items()}
        self.neurones: List[Neurone] = neurones
        nb_neurones: int = len(neurones)
        self.poids: np.ndarray = np.zeros((nb_neurones, nb_neurones), dtype=float)
        for i, connexions in connectivite.items():
            for j, poids in connexions.items():
                self.poids[i, j] = poids

        self.temps_depuis_spikes: np.ndarray = np.full(nb_neurones, np.nan, dtype=float)

        self.fonction_alpha: np.vectorize
        if fonction_alpha is None:
            self.fonction_alpha = np.vectorize(dirac)
        else:
            self.fonction_alpha = np.vectorize(fonction_alpha)

        self.update_strategy: NeuroneUpdateStrategy
        if update_strategy is None:
            self.update_strategy = EulerUpdateStrategy()
        else:
            self.update_strategy = update_strategy

    def update(self, dt: float, intensites: list[float]) -> list[bool]:

        alphas: np.ndarray = self.fonction_alpha(self.temps_depuis_spikes)
        psps: np.ndarray = alphas.dot(self.poids)
        spikes: list[bool] = []

        for i, neurone in enumerate(self.neurones):
            spikes.append(self.update_strategy.update(neurone, dt, intensites[i], psps[i]))
            self.temps_depuis_spikes[i] = 0. if spikes[i] else self.temps_depuis_spikes[i] + dt
        
        return spikes