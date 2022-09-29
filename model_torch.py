import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
from torchsummary import summary
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NN_net(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.conv1 = nn.Conv2d(1,13,3)
		#self.conv2 = nn.Conv2d(144,144,3)
		self.conv3 = nn.Conv2d(13,26,3)
		self.bn1 = nn.BatchNorm2d(13)
		self.bn2 = nn.BatchNorm2d(26)
		self.linear1 = nn.Linear(26*30*106,10)
		self.linear2 = nn.Linear(10,4)
		#self.linear3 = nn.Linear(64,4)
		self.pool = nn.MaxPool2d(2, 2)
		self.softmax = nn.Softmax(dim=1)

	def	forward(self,x):
		#print(x.shape)
		x = F.relu(self.pool(self.bn1(self.conv1(x))))
		#print(x.shape)
		#x = F.relu(self.pool(self.bn1(self.conv2(x))))
		x = F.relu(self.pool(self.bn2(self.conv3(x))))
		#print(x.shape)
		x=torch.flatten(x,1)
		x=F.relu(self.linear1(x))
		x=F.relu(self.linear2(x))
		#x=F.relu(self.linear3(x))
		x = self.softmax(x)
		return x

model = NN_net()
summary(model.to(device),(1,128,431)) 

#################################################################################################
###################         Training                ##################################
def train_one_epoch(model,dataloader,loss_fn, optimizer,device):
    correct = 0
    for data, labels in dataloader:
        data , labels = data.to(device), labels.to(device)
        #labels =labels.to(torch.float)
        #print(type(labels))
        data = data.permute(0,3,1,2)
       
        #calculate loss
        predictions = model(data)
        

        loss = loss_fn(predictions, labels)
        
        #backpropagate loss and update the weights
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _,predictions =torch.max(predictions,1) #value, index = torch.max(output,axis=1)
        correct += (predictions == labels).float().sum()
    
    accuracy = 100*correct/(293*16)
    print(f"accuracy:{accuracy}")
    print(f"loss:{loss.item()}")
    
def train(model,dataloader,loss_fn, optimizer,device,epochs):
    
    for i in range(epochs):
        print(f"epoch {i+1}/ {epochs}")
        train_one_epoch(model,dataloader,loss_fn, optimizer,device)
        print("-----------------")
    print("traning is done")
    

############################################################################################ 
######################### testing ##################################################
def evaluate(model,test_loader,ClassNames):
    correct,total=0,0
    #print(f'len of test_loader:{len(test_loader)}')
    # Deactivate autograd engine (don't compute grads since we're not training)
    
    with torch.no_grad():
        #model.eval()
        y_actual_val,y_pred_val=[],[]
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            data = data.permute(0,3,1,2)
            output=model(data)
            _,predict=torch.max(output,dim=1)
            predict = predict.detach().cpu().numpy() ## moving from gpu to cpu and tensor --> numpy
            target = target.detach().cpu().numpy()
            y_actual_val.append(target) #to convert into 1D hstack array
            y_pred_val.append(predict)
            #print(target , predict)
            total += target.shape[0]
            correct +=(predict==target).sum().item()
            #print(correct)
            
    #print(f'y_actual_list:{np.hstack(y_actual_val)}')  
          
    print('Accuracy of the network: %d %%' % (100 * correct / total))
    
    y_val = np.hstack(y_actual_val)
    y_pred_val = np.hstack(y_pred_val)
    
    conf_matrix = confusion_matrix(y_val,y_pred_val)
    print("\n\nConfusion matrix:")
    print(conf_matrix)
    
    conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
    recall_by_class = np.diagonal(conf_mat_norm_recall)
    mean_recall = np.mean(recall_by_class)

    print("Class names:", ClassNames)
    print("Per-class val acc: ",recall_by_class, "\n\n")
      
   
