from region_gaussian import RegionGaussian
from utils import flat_trans, train, test
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F 
import torch.optim as optim 
import torch.nn as nn 
import torchvision 
import torch

# Define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)   

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DRGM(nn.Module):
    def __init__(self, adopt_concat = True):
        super(DRGM, self).__init__()
        self.rg = RegionGaussian(adopt_concat = adopt_concat)
        if adopt_concat:
            self.init_size = 28*28*2
        else:
            self.init_size = 28*28
        self.fc1 = nn.Linear(self.init_size, 300)            
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)   

    def forward(self, x):
        x = self.rg(x).view(-1, self.init_size)       
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    original_loss_list = []
    drgm_loss_list = []
    drgm_cat_loss_list = []
    SoftmaxWithXent = nn.CrossEntropyLoss()

    # Define data loader
    train_loader = torch.utils.data.DataLoader(dataset = torchvision.datasets.MNIST(
        root="./mnist", train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(), 
                # transforms.Lambda(flat_trans)
        ])), batch_size=256, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root="./mnist", train=False, download=True, transform=transforms.Compose([
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
    # Deep Regional Gaussian Machine (without concat)
    # ----------------------------------------------------------------
    net = DRGM(adopt_concat = False)   
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)
    net = train(net, train_loader, optimizer, SoftmaxWithXent, drgm_loss_list)
    test(net, test_loader)

    # ----------------------------------------------------------------
    # Deep Regional Gaussian Machine (with concat)
    # ----------------------------------------------------------------
    net = DRGM(adopt_concat = True)   
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)
    net = train(net, train_loader, optimizer, SoftmaxWithXent, drgm_cat_loss_list)
    test(net, test_loader)

    # Plot
    plt.plot(range(len(original_loss_list)), original_loss_list, label = 'original network loss curve')
    plt.plot(range(len(drgm_loss_list)), drgm_loss_list, label = 'DRGM network loss curve (without concat)')
    plt.plot(range(len(drgm_cat_loss_list)), drgm_cat_loss_list, label = 'DRGM network loss curve (with concat)')
    plt.legend()
    plt.show()