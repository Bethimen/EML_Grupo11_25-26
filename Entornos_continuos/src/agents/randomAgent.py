import numpy as np


class RandomAgent:
	"""Agente aleatorio para Flappy Bird con 2 acciones discretas (0 y 1)."""

	def __init__(self, env, seed: int = None):	
		self.env = env
		self.seed = seed
		self.rng = np.random.default_rng(seed)

	def get_action(self, obs=None, evaluate: bool = False) -> int:
		"""Devuelve una acción aleatoria uniforme entre 0 y 1."""
		return int(self.rng.integers(0, 2))

	def update(self, obs, action, next_obs, reward, terminated, truncated, info=None):
		"""No hay aprendizaje en el agente aleatorio."""
		return None

	def train(self, episodes: int = 100, max_steps: int = 500):
		"""Ejecuta episodios con política aleatoria y devuelve recompensas/pasos."""
		train_rewards, train_steps = [], []

		for _ in range(episodes):
			obs, _ = self.env.reset(seed=self.seed)
			total_reward, step, done = 0.0, 0, False

			while step < max_steps and not done:
				action = self.get_action(obs)
				next_obs, reward, terminated, truncated, _ = self.env.step(action)
				self.update(obs, action, next_obs, reward, terminated, truncated)

				obs = next_obs
				total_reward += reward
				step += 1
				done = bool(terminated or truncated)

			train_rewards.append(float(total_reward))
			train_steps.append(step)

		return train_rewards, train_steps

	def evaluate(self, episodes: int = 10, max_steps: int = 500):
		"""Evalúa el agente aleatorio y devuelve recompensas por episodio."""
		rewards = []

		for _ in range(episodes):
			obs, _ = self.env.reset(seed=self.seed)
			total_reward, step, done = 0.0, 0, False

			while step < max_steps and not done:
				action = self.get_action(obs, evaluate=True)
				obs, reward, terminated, truncated, _ = self.env.step(action)
				total_reward += reward
				step += 1
				done = bool(terminated or truncated)

			rewards.append(float(total_reward))

		return rewards
