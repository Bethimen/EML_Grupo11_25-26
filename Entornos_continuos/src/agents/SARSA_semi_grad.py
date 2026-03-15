from collections import deque
from typing import Callable, Optional, Any

import numpy as np
from tqdm.auto import tqdm


class SARSASemiGradAgent:
    """
    Agente SARSA semi-gradiente n-pasos (on-policy) con aproximación lineal.


    Uso:
        agent = SARSASemiGradAgent(env)
        rewards, steps = agent.train(episodes=500)
        test_rewards = agent.evaluate(episodes=10)
    """

    def __init__(
        self,
        env,
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon_start: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,
        n_steps: int = 1,
        feature_encoder: Optional[Callable[[Any], np.ndarray]] = None,
        seed: int = None,
    ):
        if not hasattr(env.action_space, "n"):
            raise ValueError("SARSASemiGradAgent requiere action_space discreto (Discrete).")
        if n_steps < 1:
            raise ValueError("n_steps debe ser >= 1.")

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_steps = n_steps
        self.feature_encoder = feature_encoder
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        n_actions = env.action_space.n
        obs, _ = env.reset(seed=seed)
        n_features = self._encode(obs).shape[0]

        self.weights = np.zeros((n_actions, n_features), dtype=np.float32)
        self._n_step_buffer: deque = deque()



    def get_action(self, obs, evaluate: bool = False) -> int:
        """Selecciona acción epsilon-greedy; greedy puro si evaluate=True."""
        x = self._encode(obs)
        if not evaluate and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.weights.shape[0]))
        return int(np.argmax(self.weights @ x))

    def update(self, obs, action, next_obs, reward, terminated, truncated, info=None):
        """Almacena transición y aplica actualización semi-gradiente cuando corresponde."""
        done = bool(terminated or truncated)
        x = self._encode(obs)
        x_next = self._encode(next_obs)
        # Selección on-policy de la siguiente acción (SARSA)
        next_action = 0 if done else self.get_action(next_obs)

        self._n_step_buffer.append((x, action, float(reward), x_next, next_action, done))

        while (len(self._n_step_buffer) >= self.n_steps) or (done and len(self._n_step_buffer) > 0):
            x_tau, a_tau = self._n_step_buffer[0][0], self._n_step_buffer[0][1]

            G, discount, terminal_found = 0.0, 1.0, False
            for k in range(min(self.n_steps, len(self._n_step_buffer))):
                _, _, r_k, _, _, d_k = self._n_step_buffer[k]
                G += discount * r_k
                discount *= self.gamma
                if d_k:
                    terminal_found = True
                    break

            if not terminal_found and len(self._n_step_buffer) >= self.n_steps:
                x_boot = self._n_step_buffer[self.n_steps - 1][3]
                a_boot = self._n_step_buffer[self.n_steps - 1][4]
                G += (self.gamma ** self.n_steps) * float(np.dot(self.weights[a_boot], x_boot))

            td_error = G - float(np.dot(self.weights[a_tau], x_tau))
            self.weights[a_tau] += self.alpha * td_error * x_tau

            self._n_step_buffer.popleft()
            if not done:
                break

        return next_action

    def decay_epsilon(self):
        """Decae epsilon al final de cada episodio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



    def train(self, episodes: int = 1000, max_steps: int = 500):
        """
        Bucle de entrenamiento principal.

        Returns:
            train_rewards: recompensa total por episodio
            train_steps:   pasos por episodio
            train_scores:  puntuaciones por episodio
        """
        train_rewards, train_steps,train_scores = [], [], []

        for _ in tqdm(range(episodes), desc=f"SARSA semi-grad n={self.n_steps}", unit="ep"):
            obs, _ = self.env.reset(seed=self.seed)
            self._n_step_buffer.clear()

            total_reward, step, done = 0.0, 0, False
            action = self.get_action(obs)

            while step < max_steps and not done:
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                action = self.update(obs, action, next_obs, reward, terminated, truncated, info)
                obs = next_obs
                total_reward += reward
                step += 1
                done = terminated or truncated

            self.decay_epsilon()
            train_rewards.append(total_reward)
            train_steps.append(step)
            train_scores.append(info.get("score", 0))

        return train_rewards, train_steps, train_scores

    def evaluate(self, episodes: int = 10, max_steps: int = 500) -> list:
        """
        Evaluación greedy (sin exploración).

        Returns:
            Lista de recompensas totales por episodio.
        """
        rewards = []
        scores = []
        steps = []
        for _ in tqdm(range(episodes), desc="Evaluando SARSA semi-grad", unit="ep", leave=False):
            obs, _ = self.env.reset(seed=self.seed)
            total_reward, step, done = 0.0, 0, False
            while step < max_steps and not done:
                action = self.get_action(obs, evaluate=True)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                obs = next_obs
                total_reward += reward
                step += 1
                done = terminated or truncated
            steps.append(step)
            rewards.append(total_reward)
            scores.append(info.get("score", 0))
        return rewards, steps, scores



    def _encode(self, obs) -> np.ndarray:
        x = self.feature_encoder(obs) if self.feature_encoder is not None else obs
        return np.asarray(x, dtype=np.float32).reshape(-1)
