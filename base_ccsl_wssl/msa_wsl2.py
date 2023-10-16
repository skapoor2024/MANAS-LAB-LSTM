# -*- coding: utf-8 -*-
"""
Multi scale attention with within sample loss. Training on 6 hrs of IIT Madras dataset.
"""
from __future__ import division
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import pandas as pd
import glob
import random

from torch.autograd import Variable
from torch import optim, nn


########################################################################

look_back1=20 # For LSTM
look_back2=50

def lstm_data(X,f):
    X = np.array(X)
    Xdata1=[]
    Xdata2=[] 
    Ydata=[]    
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1) #so that the standard deviation never becomes zero.
    X = (X - mu) / std # normalize the data
    f1 = os.path.splitext(f)[0]  # gives first name of file without extension
    #print(f1)
    clas=f1[44:47]
    #print(clas)
                
    if (clas == 'asm'):
        Y = 0
    elif (clas == 'ben'):
        Y = 1
    elif (clas == 'guj'):
        Y = 2
    elif (clas == 'hin'):
        Y = 3
    elif (clas == 'kan'):
        Y = 4
    elif (clas == 'mal'):
        Y = 5
    elif (clas == 'man'):
        Y = 6
    elif (clas == 'odi'):
        Y = 7 
    elif (clas == 'tel'):
        Y = 8   
  
    Y=np.array([Y])
    
    for i in range(0,len(X)-look_back1,5):    #High resolution low context        
        a=X[i:(i+look_back1),:]        
        Xdata1.append(a)
    Xdata1=np.array(Xdata1)

    for i in range(0,len(X)-look_back2,10):     #Low resolution long context       
        b=X[i:(i+look_back2):3,:]        
        Xdata2.append(b)
    Xdata2=np.array(Xdata2)

    Ydata=Y
    Xdata1 = torch.from_numpy(Xdata1).float()
    Xdata2 = torch.from_numpy(Xdata2).float()
    Ydata=torch.from_numpy(Ydata).long()
    #print('The shape of data after appending look_back:', Xdata1.shape,Xdata2.shape)
    return Xdata1,Xdata2,Ydata
########################################################################

class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(80, 256,bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 64,bidirectional=True)
               
        self.fc_ha=nn.Linear(2*64,100) 
        self.fc_1= nn.Linear(100,1)           
        self.sftmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x) #input of shape (seq_len, batch, input_size): h_0 and c0 of shape (num_layers * num_directions, batch, hidden_size)
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht=torch.unsqueeze(ht, 0)        
        ha= torch.tanh(self.fc_ha(ht))
        alpha= self.fc_1(ha)
        al= self.sftmax(alpha) # Attention vector
        
        #print('ha shape',ha.shape)
        T=list(ht.shape)[1]  #T=time index
        batch_size=list(ht.shape)[0]
        D=list(ht.shape)[2]
        c=torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,D))
        #print('c size',c.size())        
        c = torch.squeeze(c,0)        
        return (c)

########################################################################
class MSA_DAT_Net(nn.Module):
    def __init__(self, model1,model2):
        super(MSA_DAT_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.classifier1 = nn.Linear(128, 128)
        self.classifier2 = nn.Linear(128, 9)
        self.sftmx = torch.nn.Softmax(dim=1)

        self.att1=nn.Linear(128,100) 
        self.att2= nn.Linear(100,1)           
        self.bsftmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x1,x2):
        u1 = self.model1(x1)
        u2 = self.model2(x2)        
        ht_u = torch.cat((u1,u2), dim=0)  # Ht  to get U vector
        #print(ht_u.size())
        ht_u = torch.unsqueeze(ht_u, 0) 
        #print(ht_u.size())
        ha_u = torch.tanh(self.att1(ht_u))
        alpha = torch.tanh(self.att2(ha_u))
        al= self.bsftmax(alpha) # Attention vector
        #print(al.size())

        Tb = list(ht_u.shape)[1]  #
        batch_size = list(ht_u.shape)[0]
        D = list(ht_u.shape)[2]
        #print(Tb,batch_size,D)
        u_vec = torch.bmm(al.view(batch_size, 1, Tb),ht_u.view(batch_size,Tb,D))
        u_vec = torch.squeeze(u_vec,0)
        
        u_vec2 = torch.tanh(self.classifier1(u_vec))
     
        ypred = self.sftmx(self.classifier2(u_vec2)) # Lang prediction
        
        return (ypred,u1,u2)     

####################################     

def train(model, criterion1, criterion2, optimizer, x_val1,x_val2, y_val):
    x1 = Variable(x_val1, requires_grad=False).cuda()
    x2 = Variable(x_val2, requires_grad=False).cuda()
    y = Variable(y_val, requires_grad=False).cuda()
    optimizer.zero_grad()

    # Forward
    fx,u1,u2 = model.forward(x1,x2)
    loss1 = criterion1(fx, y) #CE Loss
    loss2 = criterion2(u1,u2) # WSSL
    #print('Loss:',loss1,loss2)
    loss = loss1 + 0.25*abs(loss2)
    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()
    return loss.item()


model1 = LSTMNet()
model2 = LSTMNet()

model = MSA_DAT_Net(model1,model2)
model.cuda()

print(model)
    
criterion1 = torch.nn.CrossEntropyLoss(reduction='mean')
criterion2 = torch.nn.CosineSimilarity()
#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

files_list = []
folders = glob.glob('/mnt/Disk12TB/LID/IITM_IL/train_BNF6Hrs/*')  # Channel perturbed data
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)

folders = glob.glob('/mnt/Disk12TB/LID/IITM_IL/DCASEbn/bech/*')  # Channel perturbed data
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)

folders = glob.glob('/mnt/Disk12TB/LID/IITM_IL/DCASEbn/bus1/*')  # Channel perturbed data
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)

folders = glob.glob('/mnt/Disk12TB/LID/IITM_IL/DCASEbn/city/*')  # Channel perturbed data
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)

folders = glob.glob('/mnt/Disk12TB/LID/IITM_IL/DCASEbn/trai/*')  # Channel perturbed data
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)


T=len(files_list)
#print('Total Training files: ',T)
random.shuffle(files_list)


model.train()  # Traing the model
np_epoch=25
for e in range(np_epoch):    
    cost = 0.
    i=0
    j=0
    for fn in files_list:  
        df = pd.read_csv(fn,encoding='utf-16',usecols=list(range(0,80)))
        data = df.astype(np.float32)
        X = np.array(data) 
        N,D=X.shape    
        if N>look_back2:
            X1,X2,Y = lstm_data(X,fn)        
            X1 = np.swapaxes(X1, 0, 1)
            X2 = np.swapaxes(X2, 0, 1)                      
            cost += train(model, criterion1, criterion2, optimizer, X1,X2, Y)
            i=i+1       
        else:
            j=j+1
        print('msa_wsl1 on IITMadras with 4 DCASE noise  ***** Ignored:',j,'     Completed files: ', i,'/',T, 'Epoch: ',e+1,'*************  *****loss: %.4f'%(cost/i)) 
    path = "/home/administrator/Muralikrishna_H/LID/Madras_IL/wssl/withDCASE_wssl/models/msa_wsl2_e"+str(e+1)+".pth" # model with only CE loss and WSSL for LID.
    torch.save(model.state_dict(), path)

 
