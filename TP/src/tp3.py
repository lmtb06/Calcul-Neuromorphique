from neuromorphic import Reseau, Neurone, LIF, DtypeEtat, entree
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def main() -> None :
    U0: float = 0.0
    theta: float = 0.1
    C: float = 1.0
    R: float = 1.0
    intensite_cour: float = 2.2e-1

    duree: float = 20.0
    dt: float = 1.0e-3
    temps: NDArray[np.float64] = np.arange(0.0, duree, dt, dtype=np.float64)
    steps : int = temps.size

    N : int = 9

    neurones : list[Neurone] = [ 
        LIF(U0=U0, theta=theta, C=C, R=R) for _ in range(N)
    ]

    TAU: float = R * C
    def alpha(t: float) -> float:
        if np.isnan(t):
            return 0.0
        return (t / TAU) * np.exp(1 - (t / TAU))
    

    reseau : Reseau = Reseau(
        neurones, 
        {i: {j: 0.0 for j in range(N//2, N)} for i in range(N//2)},
        # fonction_alpha=alpha,
        # update_strategy=RK4UpdateStrategy()
    )


    temps_spikes_neurones : list[list[float]] = [[] for _ in range(N)]

    donnees_neurones : list[np.ndarray] = [np.empty(steps, dtype=DtypeEtat) for _ in range(N)]

    for i in range(steps):
        dt = temps[i] - (0.0 if i == 0 else temps[i-1])

        courants_entrant = [
            entree(temps[j], (j+1)%9)
            for i in range(N)
        ]

        spikes = reseau.update(dt, courants_entrant)
        for j, spike in enumerate(spikes):
            donnees_neurones[j][i] = neurones[j].vecteur_etat
            if spike:
                temps_spikes_neurones[j].append(temps[i])

    fig : Figure = plt.figure(figsize=(10, 10))
    axes : list[Axes] = fig.subplots(nrows=3, ncols=1, sharex=True)

    temps_spikes = [np.array(temps_spikes_neurones[i]) for i in range(N)]

    # Spikes en fonction du temps
    offsets = [i for i in range(0, len(temps_spikes_neurones))]
    labels = ['N {}'.format(i+1) for i in range(len(temps_spikes_neurones))]
    colors = ['C{}'.format(i) for i in range(len(temps_spikes_neurones))]
    axes[0].eventplot(temps_spikes, lineoffsets=offsets, colors=colors)
    axes[0].set_ylabel("Spikes")
    axes[0].set_yticks(offsets, labels)
    axes[0].yaxis.set_visible(False)

    # Potentiel membranaire en fonction du temps
    U_values = np.array([donnees["U"] for donnees in donnees_neurones])
    values_dict = {key: np.array([donnees[key] for donnees in donnees_neurones]) 
                  for key in ["U", "I_ext"]}
    
    U_values = values_dict["U"]
    axes[1].plot(temps, U_values.T)
    axes[1].set_ylabel("Potentiel membranaire")
    axes[1].set_ylim(U0, theta)
    axes[1].margins(y=0.1)

    # Intensité externe I_ext en fonction du temps
    I_values = values_dict["I_ext"]
    axes[2].plot(temps, I_values.T)
    axes[2].set_ylabel("Intensité externe (A)")
    axes[2].set_ylim(0)

    plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1), title="Neurones", title_fontsize='small', fontsize='small')
    plt.xlabel("Temps (ms)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()