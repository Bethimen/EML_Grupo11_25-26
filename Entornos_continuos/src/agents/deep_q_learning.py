from collections import deque
import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from gymnasium.wrappers import RecordVideo

from .components.replay_buffer import ReplayBuffer
from .components.networks import QNetwork


class DQNAgent:
    """
    Agente Deep Q-Network (DQN) para entornos Gymnasium con acción discreta.

    Implementa:
    - Experience replay buffer
    - Target network con actualización periódica
    - Double DQN (reduce sobreestimación)
    - n-step returns

    Uso:
        agent = DQNAgent(env)
        rewards, steps = agent.train(episodes=500)
        test_rewards = agent.evaluate(episodes=10)
    """

    def __init__(
        self,
        env,
        # Red neuronal
        hidden_dim: int = 64,
        num_layers: int = 3,
        # Aprendizaje
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        # Replay buffer
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        n_steps: int = 1,
        # Entrenamiento
        learning_starts: int = 1000,
        train_freq: int = 4,
        target_update_freq: int = 1000,
        tau: float = 1.0,      
        # Opciones
        double_dqn: bool = True,
        seed: int = None,
        device: torch.device = None,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.double_dqn = double_dqn
        self.seed = seed
        self.global_step = 0

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Dimensiones del espacio de estados y acciones
        n_actions = env.action_space.n
        obs, _ = env.reset(seed=seed)
        state_dim = np.asarray(obs, dtype=np.float32).reshape(-1).shape[0]

        # Redes Q y target
        self.q_net = QNetwork(state_dim, n_actions, hidden_dim, num_layers).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim, num_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(state_dim, buffer_capacity)
        self.n_step_buffer = deque(maxlen=n_steps)


    def get_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        """Selecciona acción epsilon-greedy; greedy puro si evaluate=True."""
        if not evaluate and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q_net.fc_out.out_features))
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return int(self.q_net(t).argmax().item())

    def update(self, obs, action, next_obs, reward, terminated, truncated, info=None):
        """Almacena transición y ejecuta un paso de aprendizaje."""
        done = bool(terminated or truncated)
        self.global_step += 1

        # Acumular retorno n-step
        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        if len(self.n_step_buffer) == self.n_steps or done:
            n_reward, discount = 0.0, 1.0
            for _, _, r, _, d in self.n_step_buffer:
                n_reward += discount * r
                discount *= self.gamma
                if d:
                    break
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            sn, dn = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
            self.replay_buffer.add(s0, a0, sn, n_reward, float(dn))

        # Paso de gradiente
        if (self.global_step >= self.learning_starts
                and self.global_step % self.train_freq == 0
                and len(self.replay_buffer) >= self.batch_size):
            self._learn()

        # Actualizar target network
        if self.global_step % self.target_update_freq == 0:
            self._update_target()

    def decay_epsilon(self):
        """Decae epsilon al final de cada episodio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Bucles de entrenamiento y evaluación
    # ------------------------------------------------------------------

    def train(self, episodes: int = 1000, max_steps: int = 500):
        """
        Bucle de entrenamiento principal.

        Returns:
            train_rewards: recompensa total por episodio
            train_steps:   pasos por episodio
            train_scores:  puntuaciones por episodio
        """
        train_rewards, train_steps, train_scores = [], [], []

        for _ in tqdm(range(episodes), desc=f"DQN n={self.n_steps}", unit="ep"):
            obs, _ = self.env.reset(seed=self.seed)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            self.n_step_buffer.clear()

            total_reward, step, done = 0.0, 0, False
            while step < max_steps and not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

                self.update(obs, action, next_obs, reward, terminated, truncated, info)
                obs = next_obs
                total_reward += reward
                step += 1
                done = terminated or truncated

            self.decay_epsilon()
            train_rewards.append(total_reward)
            train_steps.append(step)

        return train_rewards, train_steps,train_scores

    def evaluate(self, episodes: int = 10, max_steps: int = 500) -> list:
        """
        Evaluación greedy (sin exploración).

        Returns:
            Lista de recompensas totales por episodio.
        """
        self.q_net.eval()
        rewards = []
        steps = []
        scores = []
        for _ in tqdm(range(episodes), desc="Evaluando DQN", unit="ep", leave=False):
            obs, _ = self.env.reset(seed=self.seed)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            total_reward, step, done = 0.0, 0, False
            while step < max_steps and not done:
                action = self.get_action(obs, evaluate=True)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                total_reward += reward
                step += 1
                done = terminated or truncated
            rewards.append(total_reward)
            steps.append(step)
            scores.append(info.get("score", 0))
        self.q_net.train()
        return rewards, steps, scores

    def record_episode_video(
        self,
        video_folder: str = "videos",
        video_name: str = "dqn_eval",
        max_steps: int = 2000,
        seed: int = None,
        env_id: str = None,
        env_kwargs: dict = None,
    ) -> str:
        """
        Genera un video (.mp4) de un episodio greedy usando render_mode='rgb_array'.

        Args:
            video_folder: Carpeta de salida para el video.
            video_name: Prefijo del archivo de video.
            max_steps: Máximo de pasos del episodio.
            seed: Semilla para reset del entorno de render.
            env_id: ID del entorno (si no se pasa, intenta inferirse desde self.env.spec.id).
            env_kwargs: Argumentos extra para gym.make (por ejemplo {'use_lidar': False}).

        Returns:
            Ruta al archivo .mp4 generado.
        """
        if env_id is None:
            env_id = self.env.spec.id

        kwargs = dict(env_kwargs or {})
        render_env = gym.make(env_id, render_mode="rgb_array", **kwargs)
        os.makedirs(video_folder, exist_ok=True)

        wrapped_env = RecordVideo(
            render_env,
            video_folder=video_folder,
            episode_trigger=lambda episode_idx: True,
            name_prefix=video_name,
            disable_logger=True,
        )

        old_eps = self.epsilon
        self.epsilon = 0.0

        try:
            obs, _ = wrapped_env.reset(seed=self.seed if seed is None else seed)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)

            done = False
            step = 0
            while step < max_steps and not done:
                action = self.get_action(obs, evaluate=True)
                next_obs, _, terminated, truncated, _ = wrapped_env.step(action)
                obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                done = bool(terminated or truncated)
                step += 1
        finally:
            self.epsilon = old_eps
            wrapped_env.close()

        mp4_files = [f for f in os.listdir(video_folder) if f.startswith(video_name) and f.endswith(".mp4")]

        latest_file = max(mp4_files, key=lambda f: os.path.getmtime(os.path.join(video_folder, f)))
        return os.path.join(video_folder, latest_file)

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _learn(self):
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)
        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_net(s).gather(1, a)

        with torch.no_grad():
            if self.double_dqn:
                next_a = self.q_net(ns).argmax(1, keepdim=True)
                next_q = self.target_net(ns).gather(1, next_a)
            else:
                next_q = self.target_net(ns).max(1, keepdim=True)[0]
            target_q = r + (1 - d) * (self.gamma ** self.n_steps) * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

    def _update_target(self):
        if self.tau == 1.0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
