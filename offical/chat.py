import json
from xmlrpc.client import ResponseError
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


with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)




def search_wiki(input):
    for intent in intents["intents"]:
            if intent["tag"] == 'search wiki':
                remove_list = intent['patterns']
    Remove_list = []
    for w in remove_list:
        w = tokenize(w)
        Remove_list += w
    Remove_list = [stem(word) for word in Remove_list]
    Remove_list= set(Remove_list)
    for w in input:
        if w in Remove_list:
            input.remove(w)
    input = ' '.join(input)
    
    try:
        wiki = wk.summary(input, sentences = 3)
        return wiki
    except Exception as e: # if can't search on wiki, response to search on data saved
       result = wk.search(input, results = 5)
       result = 'do you want to khnow about: ' + ', '.join(result)
       return result

def no_response(bag_sentence):
    if np.all(bag_sentence == 0):
        return True
    else:
        return False
wiki = wk.summary('Indian people', sentences = 2)
FILE ='data.pth'
data = torch.load(FILE)

layers=data['layers'] 
all_words = data['all_words']
tags = data['tags']
responses = data["responses"]
model_state = data["model_state"]
length_tag = len(tags)
n = len(all_words)

model = Net(layers)
model.load_state_dict(model_state)
model.eval()

def Bot_responses(people):
    test = tokenize(people)  
    test = [stem(word) for word in test]
    test = exclude(test)   
    test = sorted(set(test))
    lenght_test = len(test)
    pred_list = []
    _test = test.copy()
    if no_response(bag_of_words(_test,all_words)):
        return 'I do not understand what you means'
    for t in test:
        X_test = []
        bag_test = bag_of_words(_test,all_words) 
        # print("BAG TEST",bag_test)
        bag_test.reshape(1,n)
        X_test.append(bag_test)
        # print(y_train[0])
        with torch.no_grad():
            pred = model(torch.tensor(X_test[0]).type(torch.FloatTensor))
            pred = pred.numpy()
            pred_list.append(pred)
        _test.remove(t)
    total_pred = np.array(pred_list).reshape(lenght_test,len(tags))
    total_pred = np.mean(total_pred, axis=0)
    # total_pred = solfmax_numpy(total_pred)
    tag_used = total_pred.argmax()      
    if tags[tag_used] == 'search wiki':
        answer = search_wiki(test)
    else:answer = random.choice(responses[tag_used])
    return answer
