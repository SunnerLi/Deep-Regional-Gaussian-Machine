from region_gaussian import RegionGaussian
from utils import flat_trans, train, test
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F 
import torch.optim as optim 
import torch.nn as nn 
import torchvision 
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding = 1)
        self.fc3 = nn.Linear(1280, 10)   

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 1280)
        x = self.fc3(x)
        return x

class DRGM(nn.Module):
    def __init__(self):
        super(DRGM, self).__init__()
        self.rg = RegionGaussian(region_size = 1)
        self.conv1 = nn.Conv2d(6, 10, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding = 1)
        self.fc3 = nn.Linear(1280, 10)   

    def forward(self, x):
        x = self.rg(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 1280)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    original_loss_list = []
    drgm_loss_list = []
    SoftmaxWithXent = nn.CrossEntropyLoss()

    # Define data loader
    train_loader = torch.utils.data.DataLoader(dataset = torchvision.datasets.CIFAR10(
        root="./cifar10", train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(), 
                # transforms.Lambda(flat_trans)
        ])), batch_size=256, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(
        root="./cifar10", train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(), 
                # transforms.Lambda(flat_trans)
        ])), batch_size=256, shuffle=True, num_workers=2
    )

    # ----------------------------------------------------------------
    # Traditional Network
    # ----------------------------------------------------------------
    net = Net()   
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)
    net = train(net, train_loader, optimizer, SoftmaxWithXent, original_loss_list)
    test(net, test_loader)

    # ----------------------------------------------------------------
    # Deep Regional Gaussian Machine
    # ----------------------------------------------------------------
    net = DRGM()   
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)
    net = train(net, train_loader, optimizer, SoftmaxWithXent, drgm_loss_list)
    test(net, test_loader)

    # Plot
    plt.plot(range(len(original_loss_list)), original_loss_list, label = 'original network loss curve (CNN)')
    plt.plot(range(len(drgm_loss_list)), drgm_loss_list, label = 'DRGM network loss curve (CNN)')
    plt.legend()
    plt.show()