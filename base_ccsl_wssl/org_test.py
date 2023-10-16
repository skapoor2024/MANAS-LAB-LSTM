import torch
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import os 
import numpy as np
import pandas as pd

import glob
import random

from torch.autograd import Variable
from torch.autograd import Function
from torch import optim

#import matplotlib.pyplot as plt

from rev_grad import ReverseLayerF

import sklearn.metrics

from better_lstm import LSTM

look_back = 35

def lstm_data(f):
    df = pd.read_csv(f,encoding='utf-16',usecols=list(range(0,80)))
    dt = df.astype(np.float32)
    X=np.array(dt)

    Xdt=[]
    Ydt=[]

    mu=X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std,std == 0,1)
    X = (X -mu)/std


    
    lang = f[36:39]
    #gen = f1[84]

    if (lang == 'asm'):
        Y = 0
    elif (lang == 'ben'):
        Y = 1
    elif (lang == 'guj'):
        Y = 2
    elif (lang == 'mal'):
        Y = 3
    elif (lang == 'man'):
        Y = 4
    elif (lang == 'kan'):
        Y = 5
    elif (lang == 'hin'):
        Y = 6
    elif (lang == 'tel'):
        Y = 7 
    elif (lang == 'odi'):
        Y = 8

    Ydt=np.array([Y])

    for i in range(0,len(X)-look_back,5):
        a=X[i:i+look_back,:]
        Xdt.append(a)

    Xdt=np.array(Xdt)

    Xdt=torch.from_numpy(Xdt).float()
    Ydt=torch.from_numpy(Ydt).long()

    return Xdt,Ydt
    
class LSTMNet(nn.Module):
    
    def __init__(self):
        
        super(LSTMNet,self).__init__()
        self.lstm1=LSTM(80,256,bidirectional=True,dropoutw= 0.2)
        self.lstm2=LSTM(2*256,64,bidirectional=True,dropoutw= 0.2)

        self.fc_ha=nn.Linear(128,50)
        self.fc_1=nn.Linear(50,1)
        self.smax=nn.Softmax(dim=1)

        self.class_classifier= nn.Sequential()
        self.class_classifier.add_module('fc1',nn.Linear(128,128,bias=True))
        self.class_classifier.add_module('fc2',nn.Linear(128,9,bias=True))
        #self.class_classifier.add_module('fc3',nn.Linear(128,9,bias=True))
    
    def forward(self,x):

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        ht=x[-1]
        ht=torch.unsqueeze(ht,0)

        ha=F.tanh(self.fc_ha(ht))
        alp=self.fc_1(ha)
        al = self.smax(alp)

        T = list(ha.shape)[1]
        batch_size=list(ha.shape)[0]
        c=torch.bmm(al.view(batch_size,1,T),ht.view(batch_size,T,128))
        c=torch.squeeze(c,0)

        lang_output = self.class_classifier(c)

        return lang_output
        
files_list=[]
folders = glob.glob('/home/skapr-13/Desktop/BUT/bnf1/*')
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)

A = []

for i in range(20):
    model = LSTMNet()
    path = '/home/skapr-13/Desktop/BUT/base_wts_5/e'+str(i+1)+'.pth'
    print(path)
    model.load_state_dict(torch.load(path))
    model.cuda()

    manual_seed = random.randint(1,10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    random.shuffle(files_list)
    Tru=[]
    Pred=[]

    for fn in files_list:
        X, Y = lstm_data(fn)
        X = np.swapaxes(X,0,1)

        x = Variable(X, requires_grad=True).cuda()
        o1 = model.forward(x)
        P = o1.argmax()
        P = P.cpu()

        Tru = np.append(Tru,Y)
        Pred = np.append(Pred,P)
    
    CM2=sklearn.metrics.confusion_matrix(Tru, Pred)
    print(CM2)
    acc = sklearn.metrics.accuracy_score(Tru,Pred)
    print(acc)
    A.append(acc)
print(A)
print(max(A))



