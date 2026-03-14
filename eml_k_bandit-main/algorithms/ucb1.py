"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo UCB1 para el problema de los k-brazos.
"""

import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de exploración (por defecto 1). Valores bajos -> explotación; valores altos -> exploración.
        """
        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.

        :return: índice del brazo seleccionado.
        """
        # Selecciona el brazo que no ha sido seleccionado aún
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm  

        t = np.sum(self.counts)

        ucb_values = np.zeros(self.k)

        for arm in range(self.k):
            bonus = self.c * np.sqrt(np.log(t) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        return np.argmax(ucb_values)



