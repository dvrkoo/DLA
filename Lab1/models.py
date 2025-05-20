import torch
import torch.nn as nn


# A simple MLP with configurable depth
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth, activation=nn.ReLU()):
        """
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output.
            hidden_dim (int): Dimension of hidden layers.
            depth (int): Total number of layers in the network.
            activation (nn.Module): Activation function to use.
        """
        super(SimpleMLP, self).__init__()
        if depth < 1:
            raise ValueError("Depth must be at least 1")

        layers = []
        if depth == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # First layer: input -> hidden
            layers.append(nn.Linear(input_dim, hidden_dim))
            # Middle layers: hidden -> hidden
            for _ in range(depth - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            # Final layer: hidden -> output
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # If single layer, no activation in between.
        if len(self.layers) == 1:
            return self.layers[0](x)
        # For multiple layers, apply activation between layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x


# A residual block for MLP (assumes the dimensions are the same)
class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU()):
        """
        Args:
            dim (int): The dimension of the input and output of the block.
            activation (nn.Module): Activation function to use.
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = activation

    def forward(self, x):
        x = x.view(x.size(0), -1)
        identity = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        # Add the skip connection and apply activation
        out += identity
        return self.activation(out)


# Residual MLP composed of an initial layer, several residual blocks, and an output layer
class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth, activation=nn.ReLU()):
        """
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output.
            hidden_dim (int): Dimension of hidden layers and residual blocks.
            depth (int): Number of residual blocks.
            activation (nn.Module): Activation function to use.
        """
        super(ResidualMLP, self).__init__()
        if depth < 1:
            raise ValueError("Depth must be at least 1")

        # Map input to hidden_dim space
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # A series of residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, activation) for _ in range(depth)]
        )
        # Final mapping to output
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(
        self, input_channels, num_classes, depth=2, image_size=28, base_filters=32
    ):
        """
        Args:
            input_channels (int): Number of channels in the input image (e.g., 1 for grayscale, 3 for RGB).
            num_classes (int): Number of output classes.
            depth (int): Number of convolutional blocks. Each block does conv -> ReLU -> max pooling.
            image_size (int): Height/width of the input image (assumed square).
            base_filters (int): Number of filters for the first conv layer. This number doubles at each block.
        """
        super(SimpleCNN, self).__init__()
        if depth < 1:
            raise ValueError("Depth must be at least 1")

        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        filters = base_filters

        # Build the convolutional blocks
        for i in range(depth):
            block = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.conv_blocks.append(block)
            in_channels = filters
            filters *= 2  # Increase the number of filters for the next block

        # Compute the spatial size after all pooling layers.
        final_size = image_size // (2**depth)
        self.fc = nn.Linear(in_channels * final_size * final_size, num_classes)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x
