
import json
from nlkt_utils import tokenize,stem,exclude,bag_of_words
import numpy as np
import torch 
import torch.nn as nn
import torch.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import random
import os
from model import Net, dataset,train

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
responses = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    response = intent['responses']
    tags.append(tag)
    responses.append(response)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        w = [stem(word) for word in w]
        all_words.extend(w)
        xy.append((w,tag))
all_words = exclude(all_words)
all_words = sorted(set(all_words))


def NumToVec(n, maxLength):
	vec = np.zeros(maxLength)
	vec[n] = 1
	vec = vec.reshape(1, maxLength)
	return vec

X_train = [] # m,n
y_train = [] # m,k
length_tag = len(tags)
n = len(all_words)
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    bag.reshape(1,n)
    X_train.append(bag)
    index = tags.index(tag)
    label = NumToVec(index, length_tag)
    y_train.append(label) #CrossEntropyLoss

def solfmax_torch(input):
    return  torch.exp(input)/torch.sum( torch.exp(input))
def solfmax_numpy(input):
    return np.exp(input)/np.sum(np.exp(input))

layers = [n,81,81,length_tag]
model = Net(layers)
dset = dataset(X_train,y_train)
train_loader = DataLoader(dset, batch_size= 1, shuffle= True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
Loss,Acc = train(model,criterion,train_loader, optimizer,epochs = 25)

data ={
    "model_state": model.state_dict(),
    "layers": layers,
    "all_words": all_words,
    "tags": tags,
    "responses": responses,
}
FILE = 'data.pth'
torch.save(data,FILE)
print(f'training complete .file saved to {FILE}')
