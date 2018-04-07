from torch.autograd import Variable
from tqdm import *
import numpy as np 
import torch

def flat_trans(x):
    x.resize_(28*28)
    return x
    
def train(net, loader, optimizer, criterion, loss_list, epoch = 30):
    for epoch in range(epoch):
        print("Epoch: {}".format(epoch))
        running_loss = 0.0 
        for data in tqdm(loader):
            inputs, labels = data 
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad() 
            outputs = net(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step()
            running_loss += loss.data[0]
        print('\r\rEpoch: {} | Loss: {}'.format(epoch, running_loss/2000.0))
        loss_list.append(running_loss/2000.0)
    print("Finished Training")
    return net

def test(net, loader):
    correct = 0.0 
    total = 0 
    for data in loader:
        images, labels = data 
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) 
        correct += (predicted == labels).sum()
    print("Accuracy: {}".format(correct/total))