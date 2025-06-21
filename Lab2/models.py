import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================
# Basic Models (Fixed versions)
# ============================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)

        # Better initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 1)

        # Better initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


# ============================
# Enhanced Models
# ============================
class EnhancedPolicyNet(nn.Module):
    """Enhanced policy network with better architecture and techniques"""

    def __init__(
        self, obs_dim, n_actions, hidden_size=256, num_layers=3, dropout_rate=0.1
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input normalization
        self.input_norm = nn.LayerNorm(obs_dim)

        # Hidden layers with residual connections
        layers = []
        current_dim = obs_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_size

        self.hidden_layers = nn.ModuleList(layers)

        # Output layer
        self.output = nn.Linear(hidden_size, n_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module == self.output:
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)

        # Process through hidden layers with residual connections
        residual = None
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, nn.Linear):
                if residual is not None and x.shape[-1] == residual.shape[-1]:
                    x = layer(x) + residual
                    residual = x
                else:
                    x = layer(x)
                    if i == 0:  # First layer
                        residual = x
            else:
                x = layer(x)

        # Output layer
        logits = self.output(x)
        return F.softmax(logits, dim=-1)


class EnhancedValueNet(nn.Module):
    """Enhanced value network with better architecture"""

    def __init__(self, obs_dim, hidden_size=256, num_layers=3, dropout_rate=0.1):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input normalization
        self.input_norm = nn.LayerNorm(obs_dim)

        # Hidden layers
        layers = []
        current_dim = obs_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_size

        self.hidden_layers = nn.ModuleList(layers)

        # Output layer
        self.output = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)

        # Process through hidden layers with residual connections
        residual = None
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, nn.Linear):
                if residual is not None and x.shape[-1] == residual.shape[-1]:
                    x = layer(x) + residual
                    residual = x
                else:
                    x = layer(x)
                    if i == 0:  # First layer
                        residual = x
            else:
                x = layer(x)

        # Output layer
        return self.output(x).squeeze(-1)


# ============================
# Utility Functions
# ============================
def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
