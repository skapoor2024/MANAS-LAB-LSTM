"""Contains three classification layers one for language , one for speaker and one for channel"""
#from __future__ import division
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

look_back = 35

def lstm_data(f):
    df = pd.read_csv(f,encoding='utf-16',usecols=list(range(0,80)))
    dt = df.astype(np.float32)
    X=np.array(dt)

    Xdt=[]

    mu=X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std,std == 0,1)
    X = (X -mu)/std
    f1=os.path.splitext(f)[0]
    tp = f1[35:39]
    lang = f1[48:51]
    gen = f1[51]

    if(tp == 'bnf0'):
        Y3 = 0
    elif(tp == 'bnf1'):
        Y3 = 1
    elif(tp == 'bnf2'):
        Y3 = 2

    if(lang == 'asm'):
        Y1 = 0
        if(gen == 'F'):
            Y2=0
        elif(gen == 'M'):
            Y2 =1
    elif(lang == 'ben'):
        Y1 = 1
        if(gen == 'F'):
            Y2= 2
        elif(gen == 'M'):
            Y2 = 3
    elif(lang == 'guj'):
        Y1 = 2
        if(gen == 'F'):
            Y2= 4
        elif(gen == 'M'):
            Y2 = 5
    elif(lang == 'mal'):
        Y1 = 3
        if(gen == 'F'):
            Y2= 6
        elif(gen == 'M'):
            Y2 = 7
    elif(lang == 'man'):
        Y1 = 4
        if(gen == 'F'):
            Y2= 8
        elif(gen == 'M'):
            Y2 = 9
    elif(lang == 'kan'):
        Y1 = 5
        if(gen == 'F'):
            Y2= 10
        elif(gen == 'M'):
            Y2 = 11
    elif(lang == 'hin'):
        Y1 = 6
        if(gen == 'F'):
            Y2= 12
        elif(gen == 'M'):
            Y2 = 13
    elif(lang == 'tel'):
        Y1 = 7
        if(gen == 'F'):
            Y2= 14
        elif(gen == 'M'):
            Y2 = 15
    elif(lang == 'odi'):
        Y1 = 8
        if(gen == 'F'):
            Y2= 16
        elif(gen == 'M'):
            Y2 = 17
    
    Y1=np.array([Y1])
    Y2=np.array([Y2])
    Y3=np.array([Y3])

    for i in range(0,len(X)-look_back,5):
        a=X[i:i+look_back,:]
        Xdt.append(a)

    Xdt=np.array(Xdt)

    Xdt=torch.from_numpy(Xdt).float()
    YY1=torch.from_numpy(Y1).long()
    YY2=torch.from_numpy(Y2).long()
    YY3=torch.from_numpy(Y3).long()

    return Xdt,YY1,YY2,YY3

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet,self).__init__()
        self.lstm1=nn.LSTM(80,256,bidirectional=True)
        self.lstm2=nn.LSTM(2*256,64,bidirectional=True)

        self.fc_ha=nn.Linear(128,50)
        self.fc_1=nn.Linear(50,1)
        self.smax=nn.Softmax(dim=1)

        self.lang_classifier= nn.Sequential()
        self.lang_classifier.add_module('fc1',nn.Linear(128,128,bias=True))
        self.lang_classifier.add_module('fc2',nn.Linear(128,9,bias=True))

        self.gen_classifier= nn.Sequential()
        self.gen_classifier.add_module('fd1',nn.Linear(128,128,bias=True))
        self.gen_classifier.add_module('fd2',nn.Linear(128,18,bias=True))

        self.tp_classifier = nn.Sequential()
        self.tp_classifier.add_module('ft1',nn.Linear(128,128,bias=True))
        self.tp_classifier.add_module('ft2',nn.Linear(128,3,bias=True))
    
    def forward(self,x,alpha):

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

        rev_fea = ReverseLayerF.apply(c,alpha)
        lang_output = self.lang_classifier(c)
        gen_output = self.gen_classifier(rev_fea)
        tp_output = self.tp_classifier(rev_fea)

        return lang_output,gen_output,tp_output

n_epoch = 10

files_list=[]

folders = glob.glob('/mnt/Disk12TB/shantanu_train_bnf_G/*')
for bnf in folders:
    for lang in glob.glob(bnf+'/*'):
        for f in glob.glob(lang+'/*.csv'):
            files_list.append(f)
        


l = len(files_list)
random.shuffle(files_list)

model = LSTMNet()
model.cuda()
optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum= 0.9)

loss_lang = torch.nn.CrossEntropyLoss(reduction='mean')
loss_gen = torch.nn.CrossEntropyLoss(reduction='mean')
loss_tp=torch.nn.CrossEntropyLoss(reduction='mean')

loss_lang.cuda()
loss_gen.cuda()
loss_tp.cuda()

yaxis=[]

for e in range(n_epoch):
    cost = 0.
    let = 0.
    random.shuffle(files_list)
    i=0
    for f in files_list:
        print(f)

        p = float(i+e*l)/n_epoch/l
        alpha = 2./(1. + np.exp(-5 * p)) - 1

        model.zero_grad()

        XX,YY1,YY2,YY3 = lstm_data(f)
        XNP=np.array(XX)
        if(np.isnan(np.sum(XNP))):
            continue
        
        try:
        
            XX = np.swapaxes(XX,0,1)
            X = Variable(XX, requires_grad=False).cuda()
            Y1 = Variable(YY1,requires_grad=False).cuda()
            Y2 = Variable(YY2,requires_grad=False).cuda()
            Y3 = Variable(YY3,requires_grad=False).cuda()
            fl, fg, ft = model.forward(X,alpha)
            err_l = loss_lang.forward(fl,Y1)
            err_g = loss_gen.forward(fg, Y2)
            err_t = loss_tp.forward(ft,Y3)

            err = err_l + err_g + err_t
        
            err.backward()
            optimizer.step()

            cost = cost + err.item()

            i = i+1

            let = cost/i

            print("\nloss for epoch"+str(e+1)+" completed files  "+str(i)+"/"+str(l)+"is %.8f"%(cost/i))
        
        except:
            continue
            
    yaxis.append(let)
    path = "/home/administrator/Muralikrishna_H/shantanu/org_n_9_AL_1/e"+str(e+1)+".pth"
    torch.save(model.state_dict(),path)

print(yaxis)
print(max(yaxis))