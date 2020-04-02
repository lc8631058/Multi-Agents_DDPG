from collections import namedtuple, deque
import random
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experiences", field_names=["state", "state_full", "action", "reward",
            "next_state", "next_state_full", "done"])
        self.seed = random.seed(seed)

    def add(self, state, state_full, action, reward, next_state, next_state_full, done):
        """Add a new experience to memory."""
        exp_ = self.experience(state, state_full, action, reward, next_state, next_state_full, done)
        self.memory.append(exp_)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([np.expand_dims(s_.state, 0) for s_ in samples if s_ is not None])).float().to(device)
        states_full = torch.from_numpy(np.vstack([s_.state_full for s_ in samples if s_ is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([np.expand_dims(s_.action, 0) for s_ in samples if s_ is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([s_.reward for s_ in samples if s_ is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([np.expand_dims(s_.next_state, 0) for s_ in samples if s_ is not None])).float().to(device)
        next_states_full = torch.from_numpy(np.vstack([s_.next_state_full for s_ in samples if s_ is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([s_.done for s_ in samples if s_ is not None]).astype(np.uint8)).float().to(device)
        return (states, states_full, actions, rewards, next_states, next_states_full, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)