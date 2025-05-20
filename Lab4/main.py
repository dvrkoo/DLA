import torch
import torchvision
from torchvision.datasets import FakeData
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as n
from tqdm import tqdm

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 256

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False, num_workers=8
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=8
)

fakeset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
fakeloader = torch.utils.data.DataLoader(
    fakeset, batch_size=batch_size, shuffle=False, num_workers=8
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # for data in fakeloader:
    #     x, y = data
    #     plt.imshow(x[0, :].permute(1, 2, 0))
    #     print(classes[y[0]])
    #     break

    device = "cuda" if torch.cuda.is_available() else "mps"
    model = CNN().to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 50
    for e in range(epochs):
        running_loss = 0
        for it, data in enumerate(tqdm(trainloader)):
            x, y = data
            x, y = x.to(device), y.to(device)

            yp = model(x)
            l = loss(yp, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            running_loss += l.item()
            # if it % 100 == 0:
            #    print(it, l.item())
        print(e, running_loss / len(trainloader))
