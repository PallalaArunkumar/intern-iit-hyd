import os
import torch
import math
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import librosa
from utils import PreEmphasis
import torchaudio
import pandas as pd

class xvecTDNN(nn.Module): 

    def __init__(self, num_classes=10, p_dropout=0.1):
        super(xvecTDNN, self).__init__()

        self.n_mels     = 64
        self.log_input  = True
        self.instancenorm   = nn.InstanceNorm1d(self.n_mels)
        self.torchfb        = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512, hop_length=256, window_fn=torch.hamming_window, n_mels=self.n_mels)
                )
        

        self.tdnn1 = nn.Conv1d(in_channels=self.n_mels, out_channels=128, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(128, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(256,num_classes)
        
        self.attention = nn.Sequential(
            nn.Conv1d(128,32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 128, kernel_size=1),
            nn.Softmax(dim=2),
            )

    def forward(self,x):
     
        #x  = x.reshape(-1,x.size()[-1])
        #Length = x.shape[1]
        #with torch.no_grad():
        #    with torch.cuda.amp.autocast(enabled=False):
        #   		x = self.torchfb(x)+1e-6
        #       if self.log_input: x = x.log()
        #       x = self.instancenorm(x).unsqueeze(1)
        x = x.squeeze(dim=1)
        #print(x.shape)
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))
        

        eps = 10**-5
        if self.training:
            shape = x.size()
            noise = torch.FloatTensor(shape)
            noise = noise.to("cuda")
            torch.randn(shape, out=noise)
            x += noise*eps
           
        w = self.attention(x)  # self attention 
        mu = x * w
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )       
        stats = torch.cat((mu,sg),1)       
        outp = self.fc1(stats)
        return outp

if __name__ =="__main__":
	
	if torch.cuda.is_available():
		device ='cuda'
	else:
		device = 'cpu'
		
	print(f'using {device} device ')
        
	model = xvecTDNN(10,0.1)
	summary(model.to(device),(64,626))



'''
########################################################################
#################################################################################################
###################         Training                ##################################
def train_one_epoch(model,dataloader,loss_fn, optimizer,device):
    model.train()
    correct,total = 0,0
    
    for datalabels in dataloader:
        optimizer.zero_grad()
        data=data.to(device,dtype=torch.float32)
        labels=labels.to(device,dtype=torch.long)
       
        #calculate loss
        predictions = model(data)
        

        loss = loss_fn(predictions, labels)
        
        #backpropagate loss and update the weights
        
        
        loss.backward()
        optimizer.step()
        
        _,predictions =torch.max(predictions,1) #value, index = torch.max(output,axis=1)
        total += labels.size(0)
        correct += (predictions == labels).float().sum()
    
    accuracy = 100*correct/(total)
    print(f"accuracy:{accuracy}")
    print(f"loss:{loss.item()}")
    return accuracy, loss.item()
    
    
def train(model,dataloader,model_dir,loss_fn,optimizer,device,epochs):
    
    for i in range(epochs):
        print(f"epoch {i+1}/ {epochs}")
        ACC,Loss=train_one_epoch(model,dataloader,loss_fn, optimizer,device)
        torch.save(model.state_dict(), os.path.join(model_dir, 'epoch{}_acc{}_loss{}.pth'.format(i+1,ACC,Loss)))
        print("-----------------")
    print("traning is done")'''
    


