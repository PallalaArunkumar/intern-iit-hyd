import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR,MultiStepLR
from torch.utils.data import Dataset,DataLoader
from torchsummary import summary
from tqdm.auto import tqdm
from Xvectors_model import *
from Dataset_march08 import *
from utils import *
import torch.nn.functional as F
import torch.multiprocessing as mp
from time import time




def train_step(model,train_loader,loss_fn,optimizer):
	model.train() # put model in train mode

	train_loss,train_acc = 0,0  # setup train_loss and train_acc

	#Loop through dataloader data batches
	for batch,(X,y) in enumerate(train_loader):
		X,y = X.to(device,dtype=torch.float32),y.to(device,dtype=torch.long) # Send data to target device

		#step 1 : forward pass
		y_pred = model(X)

		#step 2 : calculate the loss
		loss = loss_fn(y_pred,y,model)
		train_loss +=loss.item()

		#step 3 : set optimizer to zero grad
		optimizer.zero_grad()

		#step 4 : backpropagate loss
		loss.backward()

		#step 5 : optimizer step
		optimizer.step()
		
		#Calculate and accumulate accuracy metrics across all batches
		_,y_pred_class = torch.max(y_pred,1)
	 
		train_acc +=(y_pred_class == y).float().sum().item()/len(y_pred)

		
	#Adjust metrics to get average loss and accuracy per batch
	
	train_loss = train_loss/len(train_loader)
	train_acc = train_acc/len(train_loader)

	return train_loss,train_acc

def test_step(model,test_loader,loss_fn):

	model.eval()

	test_loss,test_acc = 0,0

	with torch.inference_mode():
		for batch,(X,y) in enumerate(test_loader):
			X,y=X.to(device,dtype=torch.float32),y.to(device,dtype=torch.long)

			y_pred = model(X)

			loss = loss_fn(y_pred,y,model)
			test_loss +=loss.item()

			_,y_pred_class = torch.max(y_pred,1)
			
			test_acc +=(y_pred_class==y).float().sum().item()/len(y_pred)
			

		test_loss = test_loss/len(test_loader)
		test_acc = test_acc/len(test_loader)

		return test_loss,test_acc

def train(model,loss_fn,optimizer,train_loader,test_loader,epochs,device,model_dir,scheduler):
	# Create empty results dictionary
	results ={"train_loss":[],
			"train_acc":[],
			"test_loss":[],
			"test_acc":[]
			}
	lrs =[]
	for epoch in tqdm(range(epochs)):
		train_loss,train_acc = train_step(model,train_loader,loss_fn,optimizer)
		lrs.append(optimizer.param_groups[0]['lr'])
		scheduler.step() ##scheduler step
	   
		test_loss,test_acc = test_step(model,test_loader,loss_fn)
		
		#torch.save(model.state_dict(),os.path.join(model_dir,f'Epoch-{epoch+1}-val_acc-{test_acc:.4f}.pth'))
			
		#print whats happeing
		print(f"Epoch:{epoch+1}|"
				f"train_Loss:{train_loss:.4f}|"
				f"train_Acc:{train_acc:.4f}|"
				f"test_Loss:{test_loss:.4f}|"
				f"test_Acc:{test_acc:.4f}")
		#update results dictionary
		results["train_loss"].append(train_loss)
		results["train_acc"].append(train_acc)
		results["test_loss"].append(test_loss)
		results["test_acc"].append(test_acc)
	
	plt.plot(range(epochs),lrs) ## ploting the learning rate
	plt.savefig('adam_0005_Multistep_lr')
	
	#np.save('fly1_multiprocessing.npy',results)
	return results

def plot_loss_curves(results,title):
	#getting the loss values from results dictionary

	loss = results['train_loss']
	test_loss = results['test_loss']

	#getting the accuracy values from the results dict
	accuracy = results['train_acc']
	test_accuracy = results['test_acc']

	epochs = range(len(results['train_loss']))

	#setup a plot
	plt.figure(figsize=(15,7))

	#plot loss
	plt.subplot(1,2,1)
	plt.plot(epochs,loss,label='train_loss')
	plt.plot(epochs,test_loss,label = 'test_loss')
	plt.xlabel('Epochs')
	plt.title('Loss')
	plt.legend()

	#plot Accuracy
	plt.subplot(1,2,2)
	plt.plot(epochs,accuracy,label='train_accuracy')
	plt.plot(epochs,test_accuracy,label='test_accuracy')
	plt.title('Accuracy')
	plt.xlabel('epochs')
	plt.legend()
	plt.savefig(title)
	
########################################################################
		
if __name__ == "__main__":
	mp.set_start_method('spawn')

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'using device:{device}')
	print(f'using Adam with lr0.0005 droupout.05 l2 regulizer with 0.001 batch size 128')

	
	TARGET_SAMPLING_RATE = 16000
	BATCH_SIZE = 128
	EPOCHS = 80
	num_workers = 25
	base_dir = '../data2020'
	train_csv = '../data2020/evalution_setup_march/train_label_clip.txt'
	valid_csv = '../data2020/evalution_setup_march/test_label_clip.txt'
	model_dir='10class_test_lr0001_d05_withaug_2'
	

	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	###################################################################################
	print("Preparing Dataset and DataLoaders:===============>\n")
	train_asc = ASC_March(base_dir,train_csv,TARGET_SAMPLING_RATE,device,data_aug=True)
	valid_asc = ASC_March(base_dir,valid_csv,TARGET_SAMPLING_RATE,device,data_aug=False) 
	
	train_itr = DataLoader(train_asc,BATCH_SIZE,shuffle=True,num_workers=num_workers)
	test_itr = DataLoader(valid_asc,BATCH_SIZE,shuffle=True,num_workers=num_workers)
	
	t_f,t_l = next(iter(train_itr))
	print('feature shape',t_f.shape)
	print('corresponding label',t_l)
	
	te_f,te_l = next(iter(test_itr))
	print('feature shape',te_f.shape)
	print('corresponding label',te_l)
	
	#################################################################################
	print("Model intilization and summary")
	#Model summary
	model = xvecTDNN(num_classes=10,p_dropout=0.05)#0.05
	summary(model.to(device),input_size=(1,160000))
	if torch.cuda.device_count() > 1:
		print("using ",torch.cuda.device_count(),"GPUs")
		model=torch.nn.DataParallel(model)
		
	model.cuda()
	
	######################################################################################
	# loss function and optimizer
	def loss_fn(output,target,model):
		loss = F.cross_entropy(output,target)
		l2_lambda = 0.001
		l2_reg = None
		for param in model.parameters():
			if l2_reg is None:
				l2_reg = param.norm(2)
			else:
				l2_reg = l2_reg + param.norm(2)
		loss = loss + l2_lambda * l2_reg
		
		return loss
	###########################################3
	Optimizer = optim.Adam(model.parameters(), lr = 0.0005)
	scheduler = MultiStepLR(Optimizer,milestones=[35,50],gamma=0.5)
	
	#######################################################################################
	#train Model
	print('Initiating Model Training: ===============>\n')
	results= train(model,loss_fn,Optimizer,train_itr,test_itr,EPOCHS,device,model_dir,scheduler)
	#np.save('plot_10sec_test_ex1_Adam0001_d05.npy',results)
	plot_loss_curves(results,'Embedding200_kerenl_64_onthe_fly_drop05_lambda001')
