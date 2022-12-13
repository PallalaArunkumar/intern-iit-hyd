import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from model_torch import *
from asc_class import *

def load_data(batch_size,train_asc,val_asc):
	return (DataLoader(train_asc,batch_size,shuffle=True)),(DataLoader(val_asc,batch_size,shuffle=True))
	
def train(model,loss_fn,optimizer,train_loader,val_loader,epochs,train_losses,valid_loss,
		model_dir,device,change_lr=None):
	
	for epoch in tqdm(range(1,epochs+1)):
		model.train()
		batch_losses=[]
		correct,y_count =0,0
		for i,data in enumerate(train_loader):
			x,y=data
			x = x.to(device,dtype=torch.float32)
			y = y.to(device,dtype=torch.float)
			#step 1 forward pass
			y_hat_logit = model(x).squeeze()
			y_pred_prob = torch.sigmoid(y_hat_logit)
			
			y_hat = torch.round(y_pred_prob)
			#print(f'y_hat_logit:{y_pred_prob}\ny_hats:{y_hat}\n targets:{y}')
			
			
			#step 2 calculate loss
			loss = loss_fn(y_hat_logit,y)
			
			batch_losses.append(loss.item())
			correct+=(y_hat==y).float().sum()
			y_count +=y.shape[0]
			
			
			
			#step 3 set optimizer to zero grad
			optimizer.zero_grad()
			
			#step 4 backpropagate the loss
			loss.backward()
			
			#step 5 optimizer step
			optimizer.step()
		
		train_losses.append(batch_losses)
		
		print(f'Epoch -{epoch} Train-Loss :{np.mean(train_losses[-1]):.5f} Train_Accuracy:{(correct/y_count):.5f}')
			
	    #################### evaluation mode ####################
			
		model.eval()
		batch_losses,batch_acc=[],[]
		y_correct,y_val_count=0,0	
		with torch.inference_mode():
			for i, data in enumerate(val_loader):
				x,y = data
				x = x.to(device,dtype=torch.float32)
				y = y.to(device,dtype=torch.float)
				#step 1 forward pass
				y_hat_logit = model(x).squeeze()
				y_pred_prob = torch.sigmoid(y_hat_logit)
				y_hat = torch.round(y_pred_prob)
				#step 2 calulate the loss
				loss = loss_fn(y_hat_logit,y)
				
				batch_losses.append(loss.item())
				y_correct+=(y_hat == y).float().sum()
				y_val_count+=y.shape[0]
				
			valid_loss.append(batch_losses)
			val_acc = (y_correct/y_val_count)
			if epoch >=160:
				torch.save(model.state_dict(),os.path.join(model_dir,f'Epoch-{epoch}_valacc-{val_acc:.5f}.pth'))
				
			print(f'Epoch -{epoch} Valid-Loss: {np.mean(valid_loss[-1]):.5f} Valid-Accuracy:{val_acc:.5f}')

if __name__ == "__main__":
	SAMPLING_RATE = 16000
	NUM_SAMPLES = SAMPLING_RATE*3
	BATCH_SIZE = 32
	EPOCHS = 250
	BASE_DIR='../data2017'
	train_csv='../data2017/evaluation_setup/uni/train_aug_2class.txt'
	val_csv='../data2017/evaluation_setup/uni/eval_2class.txt'
	pickle_dir ='featues/bi_class/'
	model_dir='model2017_2class_withaug_3sec'
	#print(os.path.isdir(pickle_dir))
	
	if not os.path.exists(pickle_dir):
		os.makedirs(pickle_dir)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	
	mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,n_fft=512,win_length=512,
	hop_length=256,window_fn=torch.hamming_window,n_mels=64 )
	
	freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
	t_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)
	
	print("Creating Training and Validation PyTorch Datasets & Batch Iterators : ===================>\n")
	tf_name = 'train2017_2class_pickle_aug_3sec.pkl'
	vl_name = 'valid2017_2class_pickle_withoutaug_3sec.pkl'
	
	if(os.path.isfile(pickle_dir+'/'+ tf_name)):
		file = open(pickle_dir+'/'+tf_name, 'rb')
		train_asc = pickle.load(file)
		        
	else:
	    train_asc = asc_dataset(train_csv,BASE_DIR,[mel_spectrogram,freq_mask,t_mask],SAMPLING_RATE,
	                            NUM_SAMPLES,device,train_=True)
	    #file_feat = open(pickle_dir+tf_name,'ab')
	    with open(pickle_dir+'/'+tf_name,'ab')as train_pickle:
	    	pickle.dump(train_asc,train_pickle)
	      

	
	print(f'total train files:{len(train_asc)}')
	
	if(os.path.isfile(pickle_dir+'/'+vl_name)):
		file = open(pickle_dir+'/'+vl_name, 'rb')
		val_asc = pickle.load(file) 
           
	else:
		val_asc = asc_dataset(val_csv,BASE_DIR,[mel_spectrogram,freq_mask,t_mask],SAMPLING_RATE,NUM_SAMPLES,device)
		with open(pickle_dir+'/'+vl_name,'ab')as val_pickle:
			pickle.dump(val_asc,val_pickle)

	print(f'total validation files:{len(val_asc)}')
	
	train_iter,val_iter = load_data(BATCH_SIZE,train_asc,val_asc)
	#print("train_iter length:",len(train_iter))
	
	
	
	#Model summary
	model = xvecTDNN(num_classes=1,p_dropout=0.2)
	summary(model.to(device),input_size=(64,626))
	model=model.to(device)
	
	
	# loss function and optimizer
	
	loss_func = nn.BCEWithLogitsLoss()
	Optimizer = torch.optim.SGD(model.parameters(), lr = 0.004,momentum=0.9)
		
	#train Model
	print('Initiating Model Training: ===============>\n')
	train_losses=[]
	val_losses=[]
	train(model,loss_func,Optimizer,train_iter,val_iter,EPOCHS,train_losses,val_losses,
	     model_dir,device)
	

				
					
					
					
				
