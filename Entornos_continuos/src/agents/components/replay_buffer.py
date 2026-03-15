import numpy as np

class ReplayBuffer:
    """
    Buffer de repetición optimizado con arrays de numpy preasignados.
    Evita los cuellos de botella de list/deque durante el muestreo.
    """
    def __init__(self, state_dim, capacity=10000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            self.rewards[idx],
            self.dones[idx]
        )
    
    def __len__(self):
        return self.size
