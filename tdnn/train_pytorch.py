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




def load_data(batch_size, train_asc, validation_asc):
	return (DataLoader(train_asc,batch_size,shuffle=True)),(DataLoader(validation_asc,batch_size,shuffle=True))

def train(model,lossfn,optimizer,train_loader,test_loader,epochs,train_losses,valid_loss,device,model_dir):
	
	for epoch in tqdm(range(1,epochs+1)):
		model.train()
		batch_losses=[]
		
		#forward and backward propagations with traindata
		for i,data in enumerate(train_loader):
			
			x,y = data
			#x=x.permute(
			#print(x.shape)
			optimizer.zero_grad()
			x = x.to(device,dtype=torch.float32)
			#print("check weater on gpu(true) or cpu(false):",x.is_cuda)
			y = y.to(device,dtype=torch.long)
			#print("label is using cuda:",x.is_cuda)
			y_hat = model(x)
			
			loss = lossfn(y_hat,y)
			loss.backward()
			batch_losses.append(loss.item())
			optimizer.step()
			
		train_losses.append(batch_losses)
		print(f'Epoch -{epoch} Train-Loss :{np.mean(train_losses[-1])}')
		
		#getting validation losses
		
		model.eval()
		batch_losses=[]
		trace_y=[]
		trace_yhat=[]
		
		#forward and backward propagation with validation data
		for i,data in enumerate(test_loader):
			x,y = data
			x = x.to(device,dtype=torch.float32)
			y = y.to(device, dtype=torch.long)
			y_hat = model(x)
			loss =lossfn(y_hat,y)
			trace_y.append(y.cpu().detach().numpy())
			trace_yhat.append(y_hat.cpu().detach().numpy())
			batch_losses.append(loss.item())
			
		valid_loss.append(batch_losses)
		trace_y =np.concatenate(trace_y)
		trace_yhat = np.concatenate(trace_yhat)
		
		#predictions & Accuracy
		predictions = trace_yhat.argmax(axis=1)
		accuracy = np.mean(predictions==trace_y)
		if epoch>=150:
			torch.save(model.state_dict(),os.path.join(model_dir,f'Epoch-{epoch}.pth'))
			
		
		#print("printing epoch number",epoch)
		
		
		print(f'Epoch -{epoch} Valid-Loss: {np.mean(valid_loss[-1])} Valid-Accuracy:{accuracy}')

##### changing learning rate funtion ######################
def setlr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']
		
def lr_decay(optimizer, epoch):
    if epoch%20==0:
    	learning_rate = get_lr(optimizer)
    	print(f'the current learning rate is:{learning_rate}')
    	new_lr = learning_rate / (10**(epoch//20))
    	optimizer = setlr(optimizer, new_lr)
    	print(f'Changed learning rate to {new_lr}')
    return optimizer
		
if __name__ == "__main__":
	SAMPLING_RATE = 16000
	NUM_SAMPLES = SAMPLING_RATE*10
	BATCH_SIZE = 32
	EPOCHS = 250
	BASE_DIR='../data2017'
	train_csv='../data2017/evaluation_setup/uni/train_aug_3class.txt'
	val_csv='../data2017/evaluation_setup/uni/eval_3class.txt'
	pickle_dir ='featues'
	model_dir='model2017_3class'
	#print(os.path.isdir(pickle_dir))
	
	if not os.path.exists(pickle_dir):
		os.mkdir(pickle_sir)
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	
	mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,n_fft=512,win_length=512,
	hop_length=256,window_fn=torch.hamming_window,n_mels=64 )
	
	freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
	t_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)
	
	print("Creating Training and Validation PyTorch Datasets & Batch Iterators : ===================>\n")
	tf_name = 'train2017_pickle_aug.pkl'
	vl_name = 'valid2017_pickle.pkl'
	
	if(os.path.isfile(pickle_dir+'/'+ tf_name)):
		file = open(pickle_dir+'/'+tf_name, 'rb')
		train_asc = pickle.load(file)
		        
	else:
	    train_asc = asc_dataset(train_csv,BASE_DIR,[mel_spectrogram,freq_mask,t_mask],SAMPLING_RATE,NUM_SAMPLES,device)
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
	
	
	
	#Model summary
	model = xvecTDNN(num_classes=3,p_dropout=0.2)
	summary(model.to(device),input_size=(64,626))
	model=model.to(device)
	
	
	# loss function and optimizer
	
	loss_func = nn.CrossEntropyLoss()
	Optimizer = torch.optim.SGD(model.parameters(), lr = 0.004,momentum=0.9)
	
	#s_dict = torch.load('models_4class_aug/Epoch-250.pth')##Loading the last epoch to train further.
	#model.load_state_dict(s_dict)
	
	#train Model
	print('Initiating Model Training: ===============>\n')
	train_losses=[]
	val_losses=[]
	train(model,loss_func,Optimizer,train_iter,val_iter,EPOCHS,train_losses,val_losses,device,model_dir)
	
	#with open('asc10.pth','wb') as f:
	#	torch.save(model,f)	
	#print(f'Trained NN is saved to asc10.pth')
 
