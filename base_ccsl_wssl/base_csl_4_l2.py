#base model with cross centroid similarity loss
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

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
    lang = f1[54:57]

    if(lang == 'asm'):
        Y1 = 0
        
    elif(lang == 'ben'):
        Y1 = 1
        
    
        
    
        
    
    elif(lang == 'kan'):
        Y1 = 2
        
    elif(lang == 'hin'):
        Y1 = 3
        
    elif(lang == 'tel'):
        Y1 = 4
        
    elif(lang == 'odi'):
        Y1 = 5
        
    
    Y11=np.array([Y1])

    for i in range(0,len(X)-look_back,5):
        a=X[i:i+look_back,:]
        Xdt.append(a)

    Xdt=np.array(Xdt)

    Xdt=torch.from_numpy(Xdt).float()
    YY1=torch.from_numpy(Y11).long()

    return Xdt,YY1,Y1

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet,self).__init__()
        self.lstm1=nn.LSTM(80,256,bidirectional=True)
        self.lstm2=nn.LSTM(2*256,64,bidirectional=True)

        self.fc_ha=nn.Linear(128,50)
        self.fc_1=nn.Linear(50,1)
        self.smax=nn.Softmax(dim=1)

        self.class_classifier= nn.Sequential()
        self.class_classifier.add_module('fc1',nn.Linear(128,6,bias=True))
    
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

        return lang_output, c
        
n_epoch = 300000
manual_seed = random.randint(1,10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

files_list=[]
folders = glob.glob('/home/iit/disk_10TB/Muralikrishna/IIITH/wssl/bnf_wssl/*')
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)

c=np.ones((6,128))      

l = len(files_list)
random.shuffle(files_list)

model = LSTMNet()
model.cuda()
optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum= 0.9)

loss_lang = torch.nn.CrossEntropyLoss(reduction='mean')
loss_lang.cuda()

c_loss = {}

for _ in range(5):
    c_loss[_] = torch.nn.CosineSimilarity()
    c_loss[_].cuda()
    
l2_loss = {}
for _ in range(5):
    l2_loss[_] = torch.nn.MSELoss()
    l2_loss[_].cuda()    
    
yaxis = []
    
for e in range(n_epoch):
    cost = 0.
    let = 0.
    random.shuffle(files_list)
    i=0
    
    my_dic = {}
    for k in range(6):
        my_dic[k] = np.empty((128,0),dtype = np.float32)
    
    for f in files_list:
        
        print(f)    
        
        df = pd.read_csv(f,encoding='utf-16',usecols=list(range(0,80)))
        data = df.astype(np.float32)
        X = np.array(data) 
        N,D=X.shape
        
        if (N>look_back):
            
            model.zero_grad()

            XX,YY1,y1 = lstm_data(f)
            XY = np.array(XX)
        
            if(np.isnan(np.sum(XY))):
                continue
            
            XX = np.swapaxes(XX,0,1)
            
            X = Variable(XX, requires_grad=False).cuda()
            Y1 = Variable(YY1,requires_grad=False).cuda()
            
            fl, u = model.forward(X)
            
            err_l = loss_lang(fl,Y1)
            
            
            u1 = u.type(torch.cuda.FloatTensor)            
            u2 = Variable(u,requires_grad = False).cpu()
            u2 = np.transpose(u2.numpy())
            my_dic[y1] = np.append(my_dic[y1],u2,axis = 1)
            
            t = []
            
            for o in range(6):
                
                tt = torch.from_numpy(c[o].reshape(1,128)).float()
                tt1 = tt.type(torch.cuda.FloatTensor)
                t.append(tt1)
            
            err_c = 0
            err_l2 = 0    
            a = 0
            f2 = 0
            
            for o in range(6):
                if(f2 == 0 and a == y1):
                    f2 = 1
                    continue
                err_c = err_c + c_loss[a](u1,t[o])
                err_l2 = err_l2 + l2_loss[a](u1,t[o])
                a = a+1
            
            err = err_l + 0.5*err_c/5 + 0.01*err_l2/5
            
            if(e<1):
            
                err = err_l
                
            if(e>1):
            
                print(err_l)
                print(err_c)
            
            err.backward()
            optimizer.step()

            cost = cost + err.item()

            i = i+1

            let = cost/i

            print("\nloss for epoch"+str(e+1)+" completed files  "+str(i)+"/"+str(l)+"is %.4f"%(cost/i))
        
    
    for i in range(6):
        ll = list(my_dic[i].shape)[1]
        c[i] = np.reshape(my_dic[i].sum(axis = 1),(1,128))
        c[i] = np.divide(c[i],ll)
        
    yaxis.append(let)
    path = "/home/iit/Muralikrishna/shantanu/base_csl_4_l2/e"+str(e+1)+".pth"
    torch.save(model.state_dict(),path)
                
                
print(yaxis)
print(min(yaxis))    
    





    
    
    
    
