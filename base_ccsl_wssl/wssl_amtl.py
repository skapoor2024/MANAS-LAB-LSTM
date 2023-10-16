# wssl with multi adverserial loss

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

from rev_grad import ReverseLayerF

look_back1=20 
look_back2=50

def lstm_data(X,f):
    X = np.array(X)
    Xdata1=[]
    Xdata2=[] 
    Ydata1 =[]
    Ydata2 = []    
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1) 
    X = (X - mu) / std 
    f1 = os.path.splitext(f)[0]  
    
    noise = f1[22:35]
    clas = f1[44:47]

    if(noise == 'org1'):
        Y2 = 0
    elif(noise == 'bech'):
        Y2 = 1
    elif(noise == 'city'):
        Y2 = 2
    elif(noise == 'trai'):
        Y2 = 3
    elif(noise == 'bus1'):
        Y2 = 4
                
    if (clas == 'asm'):
        Y1 = 0
    elif (clas == 'ben'):
        Y1 = 1
    elif (clas == 'guj'):
        Y1 = 2
    elif (clas == 'hin'):
        Y1 = 3
    elif (clas == 'kan'):
        Y1 = 4
    elif (clas == 'mal'):
        Y1 = 5
    elif (clas == 'man'):
        Y1 = 6
    elif (clas == 'odi'):
        Y1 = 7 
    elif (clas == 'tel'):
        Y1 = 8   
  
    Y1 = np.array([Y1])
    Y2 = np.array([Y2])
    
    for i in range(0,len(X)-look_back1,5):    #High resolution low context        
        a=X[i:(i+look_back1),:]        
        Xdata1.append(a)
    Xdata1=np.array(Xdata1)

    for i in range(0,len(X)-look_back2,10):     #Low resolution long context       
        b=X[i:(i+look_back2):3,:]        
        Xdata2.append(b)
    Xdata2=np.array(Xdata2)

    Ydata1 = Y1
    Ydata2 = Y2
    Xdata1 = torch.from_numpy(Xdata1).float()
    Xdata2 = torch.from_numpy(Xdata2).float()
    Ydata1 = torch.from_numpy(Ydata).long()
    Ydata2 = torch.from_numpy(Ydata2).long()
    
    return Xdata1,Xdata2,Ydata1,Ydata2

class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(80, 256,bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 64,bidirectional=True)
               
        self.fc_ha=nn.Linear(2*64,100) 
        self.fc_1= nn.Linear(100,1)           
        self.sftmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x) 
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht=torch.unsqueeze(ht, 0)        
        ha= torch.tanh(self.fc_ha(ht))
        alp= self.fc_1(ha)
        al= self.sftmax(alp) 
        
       
        T=list(ht.shape)[1]  
        batch_size=list(ht.shape)[0]
        D=list(ht.shape)[2]
        c=torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,D))        
        c = torch.squeeze(c,0)        
        return (c)

class MSA_DAT_Net(nn.Module):
    def __init__(self, model1,model2):
        super(MSA_DAT_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.att1=nn.Linear(128,100) 
        self.att2= nn.Linear(100,1)           
        self.bsftmax = torch.nn.Softmax(dim=1)

        self.lang_classifier= nn.Sequential()
        self.lang_classifier.add_module('fc1',nn.Linear(128,64,bias=True))
        self.lang_classifier.add_module('fc2',nn.Linear(64,9,bias=True))

        self.noise_classifier = nn.Sequential()
        self.noise_classifier.add_module('ft1',nn.Linear(128,64,bias=True))
        self.noise_classifier.add_module('ft2',nn.Linear(64,5,bias=True))
        
    def forward(self, x1,x2,alpha):
        u1 = self.model1(x1)
        u2 = self.model2(x2)        
        ht_u = torch.cat((u1,u2), dim=0)  
        ht_u = torch.unsqueeze(ht_u, 0) 
        ha_u = torch.tanh(self.att1(ht_u))
        alp = torch.tanh(self.att2(ha_u))
        al= self.bsftmax(alp)
        Tb = list(ht_u.shape)[1] 
        batch_size = list(ht_u.shape)[0]
        D = list(ht_u.shape)[2]
        u_vec = torch.bmm(al.view(batch_size, 1, Tb),ht_u.view(batch_size,Tb,D))
        u_vec = torch.squeeze(u_vec,0)

        rev_fea = ReverseLayerF.apply(u_vec,alpha)
        
        lang_output = self.lang_classifier(uvec)
        noise_output = self.noise_classifier(rev_fea)
        
        return (lang_output,noise_output,u1,u2)

model1 = LSTMNet()
model2 = LSTMNet()

model1.cuda()
model2.cuda()

model = MSA_DAT_Net(model1,model2)
model.cuda()
optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum= 0.9)

loss_lang = torch.nn.CrossEntropyLoss(reduction='mean')
loss_noise = torch.nn.CrossEntropyLoss(reduction='mean')
loss_emb = torch.nn.CosineSimilarity()

loss_emb.cuda()
loss_lang.cuda()
loss_noise.cuda()

yaxis = []

n_epoch = 20

files_list=[]

folders = glob.glob('/mnt/Disk12TB/shantanu_train_bnf_G/bnf0_10/*')
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)

folders = glob.glob('/mnt/Disk12TB/shantanu_train_bnf_G/bnf1_10/*')
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)
        
folders = glob.glob('/mnt/Disk12TB/shantanu_train_bnf_G/bnf2_10/*')
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)

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

        df = pd.read_csv(f,encoding='utf-16',usecols=list(range(0,80)))
        data = df.astype(np.float32)
        X = np.array(data) 
        N,D=X.shape

        if N>look_back2:

            XX1,XX2,YY1,YY2 = lstm_data(X,fn)
            XX1 = np.swapaxes(XX1,0,1)
            XX2 = np.swapaxes(XX2,0,1)
            X1 = Variable(XX1,requires_grad=False).cuda()
            Y1 = Variable(YY1,requires_grad=False).cuda()
            X2 = Variable(XX2,requires_grad=False).cuda()
            Y2 = Variable(YY2,requires_grad=False).cuda()

            fl ,fn , u1,u2 = model.forward(X1,X2,alpha)
            err_l = loss_lang(fl,Y1)
            err_n = loss_noise(fn,Y2)
            err_e = loss_emb(u1,u2)

            err = err_l + err_n + 0.25*abs(err_e)

            err.backward()
            optimizer.step()

            cost = cost + err.item()

            i = i+1

            print("\nloss for epoch"+str(e+1)+" completed files  "+str(i)+"/"+str(l)+"is %.8f"%(cost/i))

    let = cost/i
    yaxis.append(let)
    path = "/home/administrator/Muralikrishna_H/shantanu/org_3_AL_2/e"+str(e+1)+".pth"
    torch.save(model.state_dict(),path)

print(yaxis)
print(max(yaxis))


    

        

