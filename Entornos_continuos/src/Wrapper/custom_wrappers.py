import gymnasium as gym

import collections
import numpy as np

class FrameStackWrapper(gym.ObservationWrapper):
    """
    Acumula las últimas 'k' observaciones.
    """
    def __init__(self, env: gym.Env, k=3, **args):
        super().__init__(env, **args)
        self.k = k
        self.frames = collections.deque(maxlen=k)
        
        old_shape = env.observation_space.shape
        self.obs_dim = old_shape[0]
        
        low = np.repeat(env.observation_space.low, k, axis=0)
        high = np.repeat(env.observation_space.high, k, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def observation(self, obs):
        self.frames.append(obs)
        return self._get_ob()

    def _get_ob(self):
        return np.concatenate(self.frames, axis=0)