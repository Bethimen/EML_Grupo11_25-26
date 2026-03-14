"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo Softmax para el problema de los k-brazos.
"""

import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 0.1):
        """
        Inicializa el algoritmo Softmax.

        :param k: Número de brazos.
        :param tau: Parámetro de temperatura. Valores bajos -> explotación; valores altos -> exploración.
        :raises ValueError: Si tau no es positivo.
        """
        assert tau > 0, "El parámetro tau debe ser mayor que 0."
        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax.

        :return: índice del brazo seleccionado.
        """

        # Calcula las probabilidades Softmax
        exp_values = np.exp(self.values / self.tau)
        probabilities = exp_values / np.sum(exp_values)

        # Selecciona un brazo basado en las probabilidades Softmax
        chosen_arm = np.random.choice(self.k, p=probabilities)

        return chosen_arm




