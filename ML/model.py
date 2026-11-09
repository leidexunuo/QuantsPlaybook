"""两层 256 单元的 Actor-Critic 网络。"""

from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    """共享 2 x 256 主干，分别输出选股分数和状态价值。"""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, 1)
        self.value_head = nn.Linear(256, 1)
        self.cash_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observations)
        logits = self.policy_head(hidden).squeeze(-1)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values
