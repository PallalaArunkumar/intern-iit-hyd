from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pickle5 as pickle
import os
from torch.utils.data import DataLoader
from model_torch import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class asc_dataset(Dataset):
    def __init__(self,csv_file,audio_dir,feat_dir):
        self.df=pd.read_csv(csv_file,sep=',')
        self.audio_dir=audio_dir
        self.feat_dir=feat_dir
    
    def __len__(self):
        return len(self.df['scene_label'])
    
    def __getitem__(self,idx):
        audio_path=self.df.loc[idx,'filename']
        #labels=self.df['scene_label'].astype('category').cat.code.values
        labels=self.df.loc[idx,'scene_label']
        
        d={'indoor':0,'outdoor':1,'transport':2,'forest_path':3}
        #print(labels)
        for k in d.keys():
            if k==labels:
                labels=d[k]
        #print(labels)
        #print(audio_path)
 
            
        file_name=audio_path.split('/')[-1].split('.')[0]
        #print(file_name)   
        #feat_mtx=[]
        file_feat_path=self.feat_dir+'/'+file_name+'.logmel'
        #print(file_feat_path)   
        with open(file_feat_path,'rb')as f:
            temp=pickle.load(f, encoding='latin1')
                #print(temp)
            feat_matrix=np.array(temp['feat_data'])
       
        #print(feat_matrix.shape, file_name)   
        #feat_mtx = np.array(feat_mtx)
        #print(type(feat_mtx))
            
        return feat_matrix,labels
        
    
Class_Names=['Indoor','Outdoor','Transport','Forest']
                 
train_csv='../evaluation_setup/2016train_data_aug.csv'
val_csv='../evaluation_setup/eval2016_30sec.csv'
audio_directory ='data_2016/'
feat_directory='../features/logmel_10sec'   
     
training_data = asc_dataset(train_csv,audio_directory,feat_directory)

validation_data = asc_dataset(val_csv,audio_directory,feat_directory)


train_dataloader = DataLoader(training_data, batch_size=16,shuffle=True)
test_dataloader = DataLoader(validation_data,batch_size=10,shuffle=True)
'''
print(len(training_data))
print(len(validation_data))
print("----------------------------")


print(len(train_dataloader)) ##we get the no of mini batches
print(len(test_dataloader))'''

'''
for data, labels in train_dataloader:
	data , labels = data.to(device), labels.to(device)
	data = data.permute(0,3,1,2)
	print(f"data shape,{data.shape}")
	
train_features, train_labels = next(iter(train_dataloader))

print(type(train_features))
print(f"train Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

test_features , test_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {len(test_features)}")
print(f"Labels batch shape: {test_labels.size()}")

#X_train=torch.from_numpy(train_features)
#print(f'X_train type:{type(X_train)}')
#X_train = train_features.permute(0,3,1,2)'''


model = NN_net().to(device)

loss_function = nn.CrossEntropyLoss()

Optimizer = torch.optim.SGD(model.parameters(), lr = 0.004)
Epochs=10
'''
train(model,train_dataloader,loss_function,Optimizer,device,Epochs)

torch.save(model.state_dict(),"asc_model.pth") #Saving the module'''

########################### Load the model and test ################################

s_dict = torch.load("asc_model.pth")
model.load_state_dict(s_dict)
evaluate(model,test_dataloader,Class_Names) 





