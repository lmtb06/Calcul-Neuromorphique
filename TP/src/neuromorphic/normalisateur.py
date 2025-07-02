class Normalisateur:
    """
    Classe permettant de normaliser et dénormaliser des potentiels neuronaux.
    
    Cette classe convertit des potentiels membranaires entre leur représentation
    physique (en V) et une représentation normalisée entre 0 et 1.
    
    Attributes:
        U0 (float): Potentiel de repos (borne inférieure, correspond à v=0)
        theta (float): Seuil de déclenchement (borne supérieure, correspond à v=1)
        potential_range (float): Plage de potentiel (theta - U0)
    """
    
    def __init__(self, U0: float, theta: float) -> None:
        """
        Initialise un normalisateur avec les bornes de potentiel spécifiées.
        
        Args:
            U0 (float): Potentiel de repos (en V)
            theta (float): Seuil de déclenchement (en V)
        """
        self.U0: float = U0
        self.theta: float = theta
        self.potential_range: float = self.theta - self.U0

    def normaliser(self, U: float) -> float:
        """
        Normalise le potentiel U en une valeur v comprise entre 0 et 1.
        
        Args:
            U (float): Potentiel membranaire à normaliser (en V)
            
        Returns:
            float: Valeur normalisée v = (U - U0) / (theta - U0)
        """
        return (U - self.U0) / self.potential_range

    def denormaliser(self, v: float) -> float:
        """
        Retourne la valeur de potentiel U à partir de la valeur normalisée v.
        
        Args:
            v (float): Valeur normalisée entre 0 et 1
            
        Returns:
            float: Potentiel membranaire U = v * (theta - U0) + U0 (en V)
        """
        return v * self.potential_range + self.U0