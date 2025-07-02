from typing import Protocol
from .neurone import Neurone


class NeuroneUpdateStrategy(Protocol):
    @staticmethod
    def update(
        neurone: Neurone, dt: float, intensite: float, psps: float = 0.0
    ) -> bool: ...


class EulerUpdateStrategy(NeuroneUpdateStrategy):
    def __str__(self) -> str:
        return "Euler"

    def __repr__(self) -> str:
        return "EulerUpdateStrategy"

    @staticmethod
    def update(
        neurone: Neurone, dt: float, intensite: float, psps: float = 0.0
    ) -> bool:
        return neurone.updateEuler(dt, intensite, psps)


class RK4UpdateStrategy(NeuroneUpdateStrategy):
    def __str__(self) -> str:
        return "RK4"

    def __repr__(self) -> str:
        return "RK4UpdateStrategy"

    @staticmethod
    def update(
        neurone: Neurone, dt: float, intensite: float, psps: float = 0.0
    ) -> bool:
        return neurone.updateRK4(dt, intensite, psps)
