"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmNormal class for the normal distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


import numpy as np

from arms import Arm


class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución Bernoulli.

        :param p: Probabilidad de éxito.
        """
        assert 0 <= p <= 1, "p debe estar entre 0 y 1."

        self.p = p

    def pull(self):
        """
        Genera una recompensa Bernoulli.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(1, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.

        :return: Valor esperado de la distribución.
        """

        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción detallada del brazo Bernoulli.
        """
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int):
        """
        Genera k brazos con probabilidad p.

        :param k: Número de brazos a generar.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."

        # Generar k- valores únicos con p probabilidad de éxito
        p_values = np.random.uniform(0.05, 0.5, k)

        arms = [ArmBernoulli(round(p, 2)) for p in p_values]

        return arms


