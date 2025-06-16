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
from cnn import CNN
from ae import Autoencoder

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 256

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False, num_workers=8
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
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


def train_cnn(model, epochs):
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


def eval_cnn(model):
    model.eval()
    y_gt, y_pred = [], []
    for it, data in enumerate(testloader):
        x, y = data
        x, y = x.to(device), y.to(device)

        yp = model(x)

        y_pred.append(yp.argmax(1))
        y_gt.append(y)

    # Here we look at accuracy and confusion matrix
    y_pred_t = torch.cat(y_pred)
    y_gt_t = torch.cat(y_gt)

    accuracy = sum(y_pred_t == y_gt_t) / len(y_gt_t)
    print(f"Accuracy: {accuracy}")


def train_ae(model, epochs):
    for e in range(epochs):
        running_loss = 0
        for it, data in enumerate(trainloader):
            x, y = data
            x, y = x.to(device), y.to(device)

            z, x_rec = model_ae(x)
            l = mse_loss(x, x_rec)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            running_loss += l.item()
            # if it % 100 == 0:
            #    print(it, l.item())
        print(e, running_loss / len(trainloader))


def eval_ae(model):
    model_ae.eval()
    # use negative MSE since higher error means OOD
    loss = nn.MSELoss(reduction="none")

    scores_fake_ae = []
    with torch.no_grad():
        for data in fakeloader:
            x, y = data
            x = x.to(device)
            z, xr = model_ae(x)
            l = loss(x, xr)
            score = l.mean([1, 2, 3])
            scores_fake_ae.append(-score)

    scores_fake_ae = torch.cat(scores_fake_ae)

    scores_test_ae = []
    with torch.no_grad():
        for data in testloader:
            x, y = data
            x = x.to(device)
            z, xr = model_ae(x)
            l = loss(x, xr)
            score = l.mean([1, 2, 3])
            scores_test_ae.append(-score)

    scores_test_ae = torch.cat(scores_test_ae)


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
    # train_cnn(model, epochs)
    # model_path = "./model/cifar10_CNN_1.pth"
    # torch.save(model.state_dict(), model_path)
    # eval_cnn(model)

    # AE

    model_ae = Autoencoder().to(device)

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model_ae.parameters(), lr=0.0001)
    epochs = 40
    train_ae(model_ae, epochs)
    model_path = "./model/cifar10_AE_1.pth"
    torch.save(model_ae.state_dict(), model_path)
    eval_ae(model_ae)
