
# model
import numpy as np
import torch 
import torch.nn as nn
import torch.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
class Net(nn.Module):
    def __init__(self,Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size,output_size in zip(Layers[:-1],Layers[1:]):
            single_layer = nn.Linear(input_size,output_size)
            self.hidden.append(single_layer)
    def forward(self,a): # input a1
        L = len(self.hidden)
        for (l,linear_transform) in zip(range(L),self.hidden):
            if l < L-1:
                z = linear_transform(a)
                a = torch.sigmoid(z)
            else:
                output = torch.sigmoid(linear_transform(a))
        return output
class dataset():
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        return self.inputs[idx], self.labels[idx]
def train (_model, criterion, train_loader, optimizer, epochs=100):
    ACC = []
    LOSS = []
    for epoch in range(epochs):
        batch_loss = []
        correct_case = 0
        for x,y in train_loader:
            optimizer.zero_grad()
            yhat = _model(x.type(torch.FloatTensor))
            y = y.type(torch.FloatTensor)
            if yhat.argmax() == y.argmax():
                correct_case+=1
            loss = criterion(yhat, y) # tinh loss
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item()) 
        LOSS.append(np.sum(batch_loss)) #tính tổng loss của tất cả datapoint trong epoch vừa qua và append vào LOSS
        ACC.append(correct_case*100/len(train_loader))
        # print(f"Loss epoch {epoch}: {LOSS[-1]}")
        # print("Accuracy:", ACC[-1])
        # print()
    return LOSS,ACC
