#base lstm model
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
    lang = f1[61:64]

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
    
    elif(lang == 'guj'):
        Y1 = 6
        
    elif(lang == 'mal'):
        Y1 = 7
        
    
    Y1=np.array([Y1])

    for i in range(0,len(X)-look_back,5):
        a=X[i:i+look_back,:]
        Xdt.append(a)

    Xdt=np.array(Xdt)

    Xdt=torch.from_numpy(Xdt).float()
    YY1=torch.from_numpy(Y1).long()

    return Xdt,YY1

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet,self).__init__()
        self.lstm1=nn.LSTM(80,256,bidirectional=True)
        self.lstm2=nn.LSTM(2*256,64,bidirectional=True)

        self.fc_ha=nn.Linear(128,50)
        self.fc_1=nn.Linear(50,1)
        self.smax=nn.Softmax(dim=1)

        self.class_classifier= nn.Sequential()
        self.class_classifier.add_module('fc1',nn.Linear(128,128,bias=True))
        self.class_classifier.add_module('fc2',nn.Linear(128,8,bias=True))
    
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

n_epoch = 20
manual_seed = random.randint(1,10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

files_list=[]
folders = glob.glob('/home/iit/disk_10TB/Muralikrishna/wssl_wav/train_wav_bnf/*')
for folder in folders:
    for f in glob.glob(folder+'/*.csv'):
        files_list.append(f)
       

l = len(files_list)
random.shuffle(files_list)

model = LSTMNet()
model.cuda()
optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum= 0.9)

loss_lang = torch.nn.CrossEntropyLoss(reduction='mean')
loss_lang.cuda()


yaxis=[]

for e in range(n_epoch):
    cost = 0.
    let = 0.
    random.shuffle(files_list)
    i=0
    for f in files_list:
    
        print(f)
        
        df = pd.read_csv(f,encoding='utf-16',usecols=list(range(0,80)))
        data = df.astype(np.float32)
        X = np.array(data) 
        N,D=X.shape

        if N>look_back:

            model.zero_grad()

            XX,YY1 = lstm_data(f)
            XY = np.array(XX)
        
            if(np.isnan(np.sum(XY))):
                continue
            
        
        
        #try:
        
            XX = np.swapaxes(XX,0,1)
            
            X = Variable(XX, requires_grad=False).cuda()
            Y1 = Variable(YY1,requires_grad=False).cuda()

            fl = model.forward(X)
            err_l = loss_lang.forward(fl,Y1)

        
            err_l.backward()
            optimizer.step()

            cost = cost + err_l.item()

            i = i+1

            let = cost/i

            print("\nloss for epoch"+str(e+1)+" completed files  "+str(i)+"/"+str(l)+"is %.4f"%(cost/i))
            
        #except:
            #continue
    yaxis.append(let)
    path = "/home/iit/Muralikrishna/shantanu_2/base_air_2/e"+str(e+1)+".pth"
    torch.save(model.state_dict(),path)

print(yaxis)
gg = max(yaxis)
print(gg)
print(yaxis.index(gg))
