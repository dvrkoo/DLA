import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(), use_batchnorm=False):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim) if use_batchnorm else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim) if use_batchnorm else nn.Identity()
        self.activation = activation

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        return self.activation(out)


class FlexibleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        activation: nn.Module = nn.ReLU(),
        residual: bool = False,
        use_batchnorm: bool = False,
    ):
        """
        A flexible MLP supporting both standard and residual connections, with optional BatchNorm.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer size.
            depth (int): Total depth (must be >= 1).
            activation (nn.Module): Activation function.
            residual (bool): If True, use residual blocks.
            use_batchnorm (bool): If True, apply BatchNorm1d after linear layers.
        """
        super().__init__()
        if depth < 1:
            raise ValueError("Depth must be at least 1")

        self.residual = residual
        self.activation = activation
        self.use_batchnorm = use_batchnorm

        if residual:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.input_bn = (
                nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity()
            )
            self.blocks = nn.ModuleList(
                [
                    ResidualBlock(hidden_dim, activation, use_batchnorm)
                    for _ in range(depth)
                ]
            )
        else:
            self.layers = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.bns.append(
                nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity()
            )
            for _ in range(depth - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.bns.append(
                    nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity()
                )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        if self.residual:
            x = self.activation(self.input_bn(self.input_layer(x)))
            for block in self.blocks:
                x = block(x)
        else:
            for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
                x = layer(x)
                if i != len(self.layers) - 1:
                    x = self.activation(bn(x))
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, activation=nn.ReLU(), use_batchnorm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.activation = activation
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # match spatial size by pooling identity
        identity = self.pool(identity)
        out = self.pool(out)
        out += identity
        return self.activation(out)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        use_skip=True,
        use_batchnorm=True,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_batchnorm,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_skip:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        use_skip=True,
        use_batchnorm=True,
    ):
        super().__init__()
        mid_channels = out_channels
        self.use_skip = use_skip
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, bias=not use_batchnorm
        )
        self.bn1 = nn.BatchNorm2d(mid_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_batchnorm,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels) if use_batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(
            mid_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=not use_batchnorm,
        )
        self.bn3 = (
            nn.BatchNorm2d(out_channels * self.expansion)
            if use_batchnorm
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_skip:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        return self.relu(out)


class FlexibleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1000,
        block_type: str = "basic",
        layers: list = [2, 2, 2, 2],
        use_skip: bool = True,
        use_batchnorm: bool = True,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        # Select block
        if block_type == "basic":
            block = BasicBlock
        elif block_type == "bottleneck":
            block = Bottleneck
        else:
            raise ValueError("block_type must be 'basic' or 'bottleneck'")

        self.inplanes = 64
        self.use_skip = use_skip
        self.use_batchnorm = use_batchnorm

        # Initial layers
        self.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=not use_batchnorm,
        )
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize last BN in each residual block
        if zero_init_residual and use_skip:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.use_skip:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                use_skip=self.use_skip,
                use_batchnorm=self.use_batchnorm,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    use_skip=self.use_skip,
                    use_batchnorm=self.use_batchnorm,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
