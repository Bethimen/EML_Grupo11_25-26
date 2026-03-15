"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the binomial distribution arm.
"""


import numpy as np

from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de intentos.
        :param p: Probabilidad de éxito.
        """
        assert n > 0, "n debe ser mayor que 0."
        assert 0 <= p <= 1, "p debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa binomial.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.

        :return: Valor esperado de la distribución.
        """

        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 100):
        """
        Genera k brazos con diferentes probabilidades p.

        :param k: Número de brazos a generar.
        :param n: Número de intentos.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."

        # Generar k- valores únicos de n intentos con p probabilidad de éxito
        p_values = np.random.uniform(0.1, 0.9, k)

        arms = [ArmBinomial(n, round(p, 2)) for p in p_values]

        return arms


