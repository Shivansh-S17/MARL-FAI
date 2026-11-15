"""Recurrent shared actor and centralised critic (architecture fixed by the project plan)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ..env.observation import CRITIC_DIM, VEC_DIM

NEG = -1e9


class Actor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.proj = nn.Linear(32 * 400, 64)
        self.fc = nn.Linear(64 + VEC_DIM, 64)
        self.gru = nn.GRUCell(64, hidden)
        self.head = nn.Linear(hidden, 10)
        self.hidden = hidden

    def encode(self, maps, vec):
        x = Fn.relu(self.conv1(maps.float()))
        x = Fn.relu(self.conv2(x))
        x = Fn.relu(self.proj(x.flatten(1)))
        return Fn.relu(self.fc(torch.cat([x, vec], -1)))

    def step(self, maps, vec, h, mask):
        h = self.gru(self.encode(maps, vec), h)
        return self.head(h).masked_fill(~mask, NEG), h

    def sequence(self, maps, vec, h0, first, mask):
        """maps [B,T,5,20,20], vec [B,T,D], h0 [B,H], first [B,T] (reset before step), mask [B,T,10]."""
        B, T = vec.shape[:2]
        z = self.encode(maps.flatten(0, 1), vec.flatten(0, 1)).view(B, T, -1)
        h, hs = h0, []
        for t in range(T):
            h = h * (~first[:, t]).float().unsqueeze(-1)
            h = self.gru(z[:, t], h)
            hs.append(h)
        logits = self.head(torch.stack(hs, 1)).masked_fill(~mask, NEG)
        return logits


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CRITIC_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_networks(seed: int):
    """Common initial tensors: depends only on the training seed (not on the method)."""
    torch.manual_seed(int(seed))
    return Actor(), Critic()
