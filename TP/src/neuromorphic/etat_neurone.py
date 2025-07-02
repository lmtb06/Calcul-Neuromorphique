from typing import Generic, TypeVar

import numpy as np
from typing_extensions import Self

T = TypeVar("T")


def _build_dtype(names: list[str], t_types: list[type]) -> np.dtype:
    return np.dtype([(name, t_type) for name, t_type in zip(names, t_types)])


class EtatNeurone(Generic[T]):
    _fields: list[str] = ["U0", "U", "theta", "R", "C", "I_ext", "spike"]
    def __init__(self, U0: T, U: T, theta: T, R: T, C: T, I_ext: T) -> None:
        dtype: np.dtype = _build_dtype(self._fields, 
        [type(U0), type(U), type(theta), type(R), type(C), type(I_ext), type(bool)])
        self._etat: np.ndarray = np.array([(U0, U, theta, R, C, I_ext, False)], dtype=dtype)

    def __getitem__(self, key: str) -> T|bool:
        if key not in self._fields:
            raise KeyError(f"Champ '{key}' non valide.")
        return self._etat[key][-1]

    def __setitem__(self, key: str, value: T|bool) -> None:
        if key not in self._fields:
            raise KeyError(f"Champ '{key}' non valide.")
        self._etat[key][-1] = value

    def __str__(self) -> str:
        return ", ".join(f"{k}: {self._etat[k][-1]}" for k in self._fields)

    def __repr__(self) -> str:
        return f"EtatNeurone {self._etat}"

    def __copy__(self) -> Self:
        new_obj = type(self).__new__(type(self))
        new_obj._etat = self._etat.copy()
        return new_obj
    
    def _apply_op(self: Self, other, op) -> Self:
        new_obj = self.__copy__()
        for field in self._fields:
            if isinstance(other, type(self)):
                new_obj._etat[field] = op(self._etat[field], other._etat[field])
            else:
                new_obj._etat[field] = op(self._etat[field], other)
        # Reset spike
        new_obj._etat["spike"][...] = False
        
        return new_obj

    def __add__(self: Self, other: Self, /) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._apply_op(other, np.add)

    def __mul__(self: Self, other: float | int, /) -> Self:
        return self._apply_op(other, np.multiply)

    def __truediv__(self: Self, other: float | int, /) -> Self:
        return self._apply_op(other, np.divide)

E = TypeVar("E")

class DeriveeEtatNeurone(Generic[T,E]):
    _fields: list[str] = ["U0", "U", "theta", "R", "C", "I_ext"]
    def __init__(self, etatNeurone: EtatNeurone[E]) -> None:
        dtype:np.dtype = np.dtype(
            [(desc[0], desc[1]) for desc in etatNeurone._etat.dtype.descr if desc[0] in self._fields]
        )
        self._etat = np.zeros_like(etatNeurone._etat, dtype=dtype)

    def __getitem__(self, key: str) -> E:
        if key not in self._fields:
            raise KeyError(f"Champ '{key}' non valide.")
        return self._etat[key][-1]

    def __setitem__(self, key: str, value: E) -> None:
        if key not in self._fields:
            raise KeyError(f"Champ '{key}' non valide.")
        self._etat[key][-1] = value

    def __str__(self) -> str:
        return ", ".join(f"{k}: {self._etat[k][-1]}" for k in self._fields)

    def __repr__(self) -> str:
        return f"DeriveeEtatNeurone {self._etat}"
    
    def __copy__(self) -> Self:
        new_obj = type(self).__new__(type(self))
        new_obj._etat = self._etat.copy()
        return new_obj

    def _apply_op(self: Self, other, op) -> Self:
        new_derivee = self.__copy__()

        for field in self._fields:
            if isinstance(other, type(self)):
                new_derivee._etat[field] = op(self._etat[field], other._etat[field])
            else:
                new_derivee._etat[field] = op(self._etat[field], other)
        return new_derivee


    def __add__(self: Self, other: Self, /) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._apply_op(other, np.add)

    def __mul__(self: Self, other: float | int, /) -> Self:
        return self._apply_op(other, np.multiply)

    def __truediv__(self: Self, other: float | int, /) -> Self:
        return self._apply_op(other, np.divide)

    def integrer(self: Self, t: T) -> EtatNeurone[E]:
        new_etat = EtatNeurone[E](
            U0=self._etat["U0"][-1]*t,
            U=self._etat["U"][-1]*t,
            theta=self._etat["theta"][-1]*t,
            R=self._etat["R"][-1]*t,
            C=self._etat["C"][-1]*t,
            I_ext=self._etat["I_ext"][-1]*t
        )

        return new_etat

# E = TypeVar("E", bound=EtatNeurone)

class SerieEtatsNeurone(Generic[T]):
    _fields: list[str] = ["U0", "U", "theta", "R", "C", "I_ext", "spike", "t"]

    def __init__(self, steps: int, type_elems: type[T]) :
        dtype: np.dtype = _build_dtype(self._fields, 
        [type_elems, type_elems, type_elems, type_elems, type_elems, type_elems, type(bool), type_elems]
        )
        self._etats = np.empty(steps, dtype=dtype)
    
    @property
    def etats(self) -> np.ndarray:
        return self._etats
    
    def set(self, index: int, value: tuple[EtatNeurone[T], T]) -> None:
        if index < 0 or index >= len(self.etats):
            raise IndexError(f"Index '{index}' hors limites.")
        etat, t = value
        if not isinstance(etat, EtatNeurone):
            raise TypeError(f"Valeur '{etat}' doit être de type EtatNeurone.")
        
        self._etats[index] = tuple(etat[field] for field in self._fields[:-1]) + (t,)


if __name__ == "__main__":
    # Exemple d'utilisation
    etat = EtatNeurone(0.0, 0.0, 0.1, 1.0, 1.0, 0.0)
    print(f"Initial:\n{etat = }")
    etat["U"] = 1.0
    print(f"Après U = 1.0:\n{etat = }")

    etat2 = EtatNeurone(0.0, 0.0, 0.1, 1.0, 1.0, 0.0)
    etat3 = etat + etat2
    print(f"Après addition:\n{etat3 = }")

    derivee = DeriveeEtatNeurone(etat)
    print(f"Après dérivée:\n{derivee = }")
    derivee["U"] = 0.5
    print(f"Après dérivée U = 0.5:\n{derivee = }")

    derivee2 = DeriveeEtatNeurone(etat2)
    derivee2["U"] = 0.3
    derivee3 = derivee + derivee2
    print(f"Après addition de dérivées:\n{derivee3 = }")

    etat4 = derivee3.integrer(0.1)
    print(f"Après intégration:\n{etat4 = }")

    series = SerieEtatsNeurone(10, float)
    print(series.etats)
    series.set(0, (etat, 0.0))
    print(series.etats[0])
    series.set(1, (etat2, 0.1))
    print(series.etats[1])
