import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import librosa
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import log_loss
from train_pytorch import *
from model_torch import *
from asc_class import *
import seaborn as sns

def load_test_data(batch_size,test_csv):
	return (DataLoader(test_csv,batch_size,shuffle=True))
	

def inference_mode(model,test_loader,device):
	model.eval()
	correct,y_count=0,0
	with torch.inference_mode():
		y_actual_val,y_pred_val=[],[]  ##lists to save target and predicted values
		for i,data in enumerate(test_loader):
			x,y = data
			x = x.to(device,dtype=torch.float32)
			y = y.to(device,dtype=torch.float)
			
			# forward pass
			y_hat_logit = model(x).squeeze()
			y_pred_prob = torch.sigmoid(y_hat_logit)
			y_hat = torch.round(y_pred_prob)
			
			correct+=(y_hat == y).float().sum()
			y_count+=y.shape[0]
			
			# detaching the values from the gpu and appending them into list
			y_hat = y_hat.detach().cpu().numpy()
			y_pred_val.append(y_hat)
			y = y.detach().cpu().numpy()
			y_actual_val.append(y)
			
	acc = (correct/y_count)
	print(f'Accuracy of the inference Mode:{acc:.3f}')
	
	y_val = np.hstack(y_actual_val)
	y_predicted_val = np.hstack(y_pred_val)
	
	conf_matrix = confusion_matrix(y_val,y_predicted_val)
	print('\n\n Confusion matrix')
	print(conf_matrix)
	#saving the conf_matrix as numpy file
	#np.save('biclass.npy',conf_matrix)
	
	conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
	recall_by_class = np.diagonal(conf_mat_norm_recall)
	mean_recall = np.mean(recall_by_class)

    #print("Class names:", ClassNames)
	print("Per-class val acc: ",recall_by_class, "\n\n")

if __name__ == "__main__":
	SAMPLING_RATE=16000
	NUM_SAMPLES = 16000*10
	BATCH_SIZE = 32
	
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
		
	print(f'using {device} device')
	
	
	mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,n_fft=512,win_length=512,
	hop_length=256,window_fn=torch.hamming_window,n_mels=64 )
	
	freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
	t_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)
	

	te_csv='hundred_clips.txt'
	
	BASE_DIR='../test2017'
	pickle_dir='featues'
	

	te_name = 'hundred_clips.pkl'


	if(os.path.isfile(pickle_dir+'/'+ te_name)):
		file = open(pickle_dir+'/'+te_name, 'rb')
		test_asc = pickle.load(file)
		        
	else:
	    test_asc = asc_dataset(te_csv,BASE_DIR,[mel_spectrogram,freq_mask,t_mask],SAMPLING_RATE,NUM_SAMPLES,device)
	    
	    with open(pickle_dir+'/'+te_name,'ab')as test_pickle:
	    	pickle.dump(test_asc,test_pickle)

	print(f'total test files:{len(test_asc)}')
	
	test_iter = load_test_data(BATCH_SIZE,test_asc)
	'''
	model = xvecTDNN(1,0).to(device)
	#loading the model
	m='Epoch-199_valacc-0.97206.pth'
	print(m)
	s_dict = torch.load('model2017_2class_withaug_3sec/'+m)
	model.load_state_dict(s_dict)
	print(inference_mode(model,test_iter,device))'''
			
			
			
			
