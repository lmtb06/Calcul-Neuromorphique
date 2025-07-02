from typing import Protocol, TypeVar

from numpy import float64
from numpy.typing import NDArray
from .neurone import Neurone
from .reseau import Reseau
from typing import Union

T = TypeVar("T", bound=Union[Neurone, Reseau])


class Enregistrement(Protocol[T]):
    def ajouter(self, element: T) -> None: ...

    def __getitem__(self, key: str) -> NDArray[float64]: ...
