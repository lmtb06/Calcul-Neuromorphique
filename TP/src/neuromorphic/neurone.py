import copy
from abc import ABC, abstractmethod
from typing import Optional, TypeVar


from .etat_neurone import DeriveeEtatNeurone, EtatNeurone
from .integrateur import RK4, Euler, Integrateur


class Neurone(ABC):
    def __init__(self, etat: EtatNeurone) -> None:
        self._etat: EtatNeurone = etat
        self._etat_initial: EtatNeurone = copy.copy(etat)

    @staticmethod
    @abstractmethod
    def _fonction_derivatrice(t: float, y: EtatNeurone) -> DeriveeEtatNeurone: ...

    def _check_and_emit_spike(self) -> bool:
        """Vérifie si le neurone émet un spike et le réinitialise si besoin."""
        if self._etat["U"] >= self._etat["theta"]:
            self._etat["U"] = self._etat["U0"]
            self._etat["spike"] = True
        return self._etat["spike"]

    def _add_psps(self, total_psp: float) -> None:
        """Ajoute les potentiels post-synaptiques (PSPs)."""
        self._etat["U"] += total_psp

    def _update(
        self,
        dt: float,
        I_ext: float,
        psps: float,
        integrateur: type[Integrateur],
    ) -> bool:
        self._etat["I_ext"] = I_ext
        self._etat["spike"] = False

        y0: EtatNeurone = self._etat
        f = self._fonction_derivatrice

        self._etat = integrateur.step(fonction=f, dt=dt, t0=0.0, y0=y0)

        self._add_psps(psps)

        return self._check_and_emit_spike()

    def updateEuler(self, dt: float, I_ext: float, psps: float = 0.0) -> bool:
        """Met à jour l'état en utilisant l'intégrateur d'Euler."""
        return self._update(dt, I_ext, psps, integrateur=Euler)

    def updateRK4(self, dt: float, I_ext: float, psps: float = 0.0) -> bool:
        """Met à jour l'état en utilisant l'intégrateur de Runge-Kutta 4."""
        return self._update(dt, I_ext, psps, integrateur=RK4)

    def reset(self) -> None:
        """Réinitialise l'état du neurone à son état initial."""
        self._etat = copy.copy(self._etat_initial)

    @property
    def etat(self) -> EtatNeurone[float]:
        return copy.copy(self._etat)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {{ {self._etat} }}"


class LIF(Neurone):
    def __init__(
        self,
        U0: float = 0.0,
        U: Optional[float] = None,
        theta: float = 0.1,
        R: float = 1.0,
        C: float = 1.0,
        I_ext: float = 0.0,
    ) -> None:
        U = U if U is not None else U0
        etat = EtatNeurone[float](U0=U0, U=U, theta=theta, R=R, C=C, I_ext=I_ext)
        super().__init__(etat)

    @staticmethod
    def _fonction_derivatrice(t: float, y: EtatNeurone) -> DeriveeEtatNeurone:
        """Calcul de la dérivée pour le modèle LIF."""

        U0 = y["U0"]
        U = y["U"]
        R = y["R"]
        C = y["C"]
        I_ext = y["I_ext"]

        tau = R * C
        diff_potentiel = U - U0

        derivee = DeriveeEtatNeurone(y)
        derivee["U"] = (R * I_ext - diff_potentiel) / tau

        return derivee

if __name__ == "__main__":
    # Exemple d'utilisation
    neurone1 = LIF()
    neurone2 = LIF()
    # print(neurone1)
    # neurone1.updateEuler(dt=0.1, I_ext=1.0)
    # print(neurone1)
    # neurone1.updateRK4(dt=0.1, I_ext=1.0)
    # print(neurone1)
    # neurone1.reset()
    # print(neurone1)

    # from .etat_neurone import SerieEtatsNeurone
    # etat_ts = SerieEtatsNeurone(steps=3, type_elems=float)
    # etat_ts.set(0, (neurone1.etat, 0.0))
    # neurone1.updateEuler(dt=0.1, I_ext=1.0)
    # etat_ts.set(1, (neurone1.etat, 0.1))
    # neurone1.updateRK4(dt=0.1, I_ext=1.0)
    # etat_ts.set(2, (neurone1.etat, 0.2))
    # print(etat_ts.etats)
    dt: float = 1.0e-2

    from time import perf_counter
    start_time = perf_counter()

    for i in range(1_000_000):
        # print(f"Neurone Euler: \n\t{neurone1.etat}")
        # print(f"Neurone RK4: \n\t{neurone2.etat}")
        neurone1.updateEuler(dt=dt, I_ext=1.0)
        neurone1.updateRK4(dt=dt, I_ext=1.0)

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"L'opération a pris {duration:.4f} secondes.")