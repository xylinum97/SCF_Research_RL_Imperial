"""Replay buffer + Ornstein-Uhlenbeck exploration noise (config-independent)."""
import random
import numpy as np
import torch

from .util import DEVICE


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.cap = capacity; self.buf = []; self.pos = 0

    def add(self, s, a, r, s2, d):
        data = (s, a, r, s2, float(d))
        if len(self.buf) < self.cap:
            self.buf.append(data)
        else:
            self.buf[self.pos] = data
        self.pos = (self.pos + 1) % self.cap

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.tensor(s,  dtype=torch.float32, device=DEVICE),
                torch.tensor(a,  dtype=torch.float32, device=DEVICE),
                torch.tensor(r,  dtype=torch.float32, device=DEVICE).unsqueeze(1),
                torch.tensor(s2, dtype=torch.float32, device=DEVICE),
                torch.tensor(d,  dtype=torch.float32, device=DEVICE).unsqueeze(1))

    def __len__(self):
        return len(self.buf)


class OUNoise:
    def __init__(self, dim=3, mu=0.0, theta=0.15, sigma=0.25, sigma_min=0.05, decay=0.9998):
        self.dim = dim; self.mu = mu; self.theta = theta
        self.sigma = sigma; self.sigma_min = sigma_min; self.decay = decay
        self.reset()

    def reset(self):
        self.state = np.ones(self.dim, dtype=np.float32) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.dim)
        self.state = self.state + dx.astype(np.float32)
        return self.state

    def step(self):
        self.sigma = max(self.sigma_min, self.sigma * self.decay)

    def boost(self, sigma):
        self.sigma = sigma; self.reset()
