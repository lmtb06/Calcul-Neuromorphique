from dataclasses import dataclass, replace
from enum import Enum

@dataclass
class DonneesNeurone:
    potentielMembranaireRepos: float
    potentielMembranaire: float
    seuilSpike: float
    capacite: float
    resistance: float
    courantEntrant: float
    picoSecondesDepuisDernierSpike: int
    spike: bool

class TypeTravailleur(Enum):
    PROCESSEUR_NEURONE = 0

class ProcesseurNeuroneLIF:
    dn: DonneesNeurone
    deltaPicoSeconde: int

    def assignerDonnees(self, dn:DonneesNeurone, deltaPicoSeconde: int):
        self.dn = dn
        self.deltaPicoSeconde = deltaPicoSeconde
        self.dn.spike = False
        # TODO Faire en sorte que le nb de pico seconde ecoulé ne soit pas incrémenté une fois qu'on dépasse le seuil pour lequel il a de l'influence (se baser sur R * C (Tau))
        if self.dn.picoSecondesDepuisDernierSpike is not None:
            self.dn.picoSecondesDepuisDernierSpike += deltaPicoSeconde

    def deltaPotentiel(dn: DonneesNeurone) -> float:
        tau = (dn.resistance * dn.capacite)
        diffPotentiel = dn.potentielMembranaire - \
            dn.potentielMembranaireRepos
        delta = ((dn.resistance *
                 dn.courantEntrant) - diffPotentiel) / tau
        return delta

    def processSpikeBehavior(dn: DonneesNeurone) -> None:
        if dn.potentielMembranaire > dn.seuilSpike:
            dn.spike = True
            dn.picoSecondesDepuisDernierSpike=0
            dn.potentielMembranaire = dn.potentielMembranaireRepos

    def stepEuler(self) -> None:
        ProcesseurNeuroneLIF.processSpikeBehavior(self.dn)

        self.dn.potentielMembranaire += (self.deltaPicoSeconde/1e-12) * \
            ProcesseurNeuroneLIF.deltaPotentiel(self.dn)

    def stepRK4(self) -> None:
        ProcesseurNeuroneLIF.processSpikeBehavior(self.dn)
        tempNeuronData = replace(self.dn)
        k1 = ProcesseurNeuroneLIF.deltaPotentiel(self.dn)
        tempNeuronData.potentielMembranaire += ((self.deltaPicoSeconde/1e-12) * k1)/2
        k2 = ProcesseurNeuroneLIF.deltaPotentiel(tempNeuronData)
        tempNeuronData.potentielMembranaire = self.dn.potentielMembranaire + \
            ((self.deltaPicoSeconde/1e-12) * k2)/2
        k3 = ProcesseurNeuroneLIF.deltaPotentiel(tempNeuronData)
        tempNeuronData.potentielMembranaire = self.dn.potentielMembranaire + \
            ((self.deltaPicoSeconde/1e-12) * k3)
        k4 = ProcesseurNeuroneLIF.deltaPotentiel(tempNeuronData)
        self.dn.potentielMembranaire += ((self.deltaPicoSeconde/1e-12) / 6) * \
            (k1 + 2*k2 + 2*k3 + k4)


if __name__ == "__main__":
    neuronData1 = DonneesNeurone(
        potentielMembranaireRepos=0.0,
        potentielMembranaire=0.0,
        seuilSpike=1.0e-1,
        capacite=1.0,
        resistance=1.0,
        courantEntrant=1.0,
        picoSecondesDepuisDernierSpike=None,
        spike=False
    )
    neuronData2 = DonneesNeurone(
        potentielMembranaireRepos=0.0,
        potentielMembranaire=0.0,
        seuilSpike=1.0e-1,
        capacite=1.0,
        resistance=1.0,
        courantEntrant=1.0,
        picoSecondesDepuisDernierSpike=None,
        spike=False
    )
    dt: float = 1.0e-2 * 1e12

    processor1 = ProcesseurNeuroneLIF()
    processor1.assignerDonnees(neuronData1, dt)
    processor2 = ProcesseurNeuroneLIF()
    processor2.assignerDonnees(neuronData2, dt)

    from time import perf_counter
    start_time = perf_counter()

    for i in range(1_000_000):
        # print(f"Neurone Euler: \n\t{neuronData1}")
        # print(f"Neurone RK4: \n\t{neuronData2}")
        processor1.stepEuler()
        processor2.stepRK4()

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"L'opération a pris {duration:.4f} secondes.")
