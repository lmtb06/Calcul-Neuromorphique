from typing import Callable, Generic, TypeVar, Protocol, runtime_checkable
from typing_extensions import Self

Z1 = float | int


@runtime_checkable
class Temps(Protocol):
    def __add__(self: Self, other: Self, /) -> Self: ...
    def __mul__(self: Self, other: Z1, /) -> Self: ...
    def __truediv__(self: Self, other: Z1, /) -> Self: ...


@runtime_checkable
class Etat(Protocol):
    def __add__(self: Self, other: Self, /) -> Self: ...
    def __mul__(self: Self, other: Z1, /) -> Self: ...
    def __truediv__(self: Self, other: Z1, /) -> Self: ...


T = TypeVar("T", bound=Temps)
E = TypeVar("E", bound=Etat)


@runtime_checkable
class Derivee(Protocol[T, E]):
    def __add__(self: Self, other: Self, /) -> Self: ...
    def __mul__(self: Self, other: Z1, /) -> Self: ...
    def __truediv__(self: Self, other: Z1, /) -> Self: ...
    def integrer(self: Self, other: T, /) -> E: ...


class Integrateur(Protocol[T, E]):
    @staticmethod
    def step(
        fonction: Callable[[T, E], Derivee[T, E]], dt: T, t0: T, y0: E
    ) -> E: ...


class Euler(Generic[T, E]):
    @staticmethod
    def step(fonction: Callable[[T, E], Derivee[T, E]], dt: T, t0: T, y0: E) -> E:
        y_prime = fonction(t0, y0)
        return y0 + (y_prime.integrer(dt))


class RK4(Generic[T, E]):
    @staticmethod
    def step(fonction: Callable[[T, E], Derivee[T, E]], dt: T, t0: T, y0: E) -> E:
        k1 = fonction(t0, y0)
        k2 = fonction(t0 + (dt / 2), y0 + k1.integrer(dt / 2))
        k3 = fonction(t0 + (dt / 2), y0 + k2.integrer(dt / 2))
        k4 = fonction(t0 + dt, y0 + k3.integrer(dt))
        y_prime = (k1 + k2 * 2 + k3 * 2 + k4) / 6
        return y0 + y_prime.integrer(dt)
