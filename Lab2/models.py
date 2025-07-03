import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNet(nn.Module):
    def __init__(
        self, obs_dim, n_actions, hidden_size=128, num_layers=2, activation=nn.ReLU
    ):
        """
        Flexible Policy Network.

        Args:
            obs_dim (int): Dimension of the observation space.
            n_actions (int): Number of possible actions.
            hidden_size (int): Number of neurons in each hidden layer.
            num_layers (int): Number of hidden layers.
            activation (nn.Module): The activation function to use (e.g., nn.ReLU, nn.Tanh).
        """
        super().__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(obs_dim, hidden_size))
        layers.append(activation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_size, n_actions))

        # Combine all layers into a sequential model
        self.net = nn.Sequential(*layers)

        # Apply your custom weight initialization
        self.init_weights()

    def init_weights(self):
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                # Use smaller gain for the final layer before softmax
                if i == len(self.net) - 1:
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # The sequential model handles the main forward pass
        logits = self.net(x)
        # Apply softmax to get action probabilities
        return F.softmax(logits, dim=-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_size=128, num_layers=2, activation=nn.ReLU):
        """
        Flexible Value Network.

        Args:
            obs_dim (int): Dimension of the observation space.
            hidden_size (int): Number of neurons in each hidden layer.
            num_layers (int): Number of hidden layers.
            activation (nn.Module): The activation function to use.
        """
        super().__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(obs_dim, hidden_size))
        layers.append(activation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())

        # Output layer (outputs a single value)
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

        # Apply your custom weight initialization
        self.init_weights()

    def init_weights(self):
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                # Use gain=1.0 for the final value layer
                if i == len(self.net) - 1:
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # Return the raw value. Squeezing is best done in the training loop
        # to avoid issues with batch size 1.
        return self.net(x)
