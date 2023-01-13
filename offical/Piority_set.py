import json
import wikipedia as wk
from nlkt_utils import tokenize,stem,exclude,bag_of_words
import numpy as np
import torch 
import torch.nn as nn
import torch.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import random
import os
from model import Net

#Piority set

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
