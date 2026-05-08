"""Neural network modules for DQN agents."""
import torch
import torch.nn as nn
from typing import Sequence


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations."""

    def __init__(self, input_size: int, output_size: int, hidden_sizes: Sequence[int] = (128, 128)):
        """
        Args:
            input_size: input dimension
            output_size: output dimension (number of actions)
            hidden_sizes: sizes of hidden layers
        """
        super().__init__()
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
