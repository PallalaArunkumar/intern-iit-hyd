import os
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn import preprocessing
from tqdm import tqdm
import pickle5 as pickle
from train_pytorch import *



class asc_dataset(Dataset):

    def __init__(self,csv_file,base_dir,transformation,target_sample_rate,num_samples,device):
        self.df=pd.read_csv(csv_file,sep=' ',header=None)
        #print(len(self.df))
        self.data=[]
        self.labels=[]
        self.c2i={}
        self.i2c={}
        self.categories=sorted(self.df[1].unique())
        
        self.device=device
        self.transformation = transformation[0].to(self.device)
        self.transformation1 = transformation[1].to(self.device)
        self.transformation2 = transformation[2].to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        for i, category in enumerate(self.categories):
        	self.c2i[category]=i
        	self.i2c[i]=category
        print(self.c2i)
        #print(self.i2c)
        	
        for ind in tqdm(range(len(self.df))):
        	row = self.df.iloc[ind]
        	file_path = os.path.join(base_dir,row[0])
        
        	signal,sr = torchaudio.load(file_path)
        	signal = signal.to(self.device)
        
        	# resampling to brinig uniformity
        	signal = self._resample_if_necessary(signal,sr)
        
        	#data cleaning to convert all wav files into monoaudio
        	signal = self._mix_down_if_necessary(signal)
        
        	#processing Audio of different Length
        	signal = self._cut_if_necessary(signal)
        	signal = self._right_pad_if_necessary(signal)
        	
        	#applying transform
        	signal = self.transformation(signal)
        	#print(signal.shape)
        	signal = signal.log()
        	
        	signal = self.transformation1(signal) # frequency masking
        	
        	signal = self.transformation2(signal) # Time and Frequency both masked
        
        	self.data.append(signal)
        	self.labels.append(self.c2i[row[1]])
        
    	
    def __len__(self):
    	return (len(self.df))
    
    def __getitem__(self,index):
    	
    	return self.data[index],self.labels[index]
    
    def _resample_if_necessary(self,signal,sr):
    	if sr != self.target_sample_rate:
    		resampler = torchaudio.transforms.Resample(sr,self.target_sample_rate)
    		resampler = resampler.to(self.device)
    		signal = resampler(signal)
    	return signal
    
    def _mix_down_if_necessary(self,signal):
    	## no. of channels : signal.shape[0]
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal 
        
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    
        
        
if __name__ == "__main__":
	BASE_DIR='../data'
	CSV_FILE = BASE_DIR+'/evaluation_setup/train_80per_20.txt'
	val_csv='../data/evaluation_setup/val_20per_20.txt'
	SAMPLING_RATE = 16000
	NUM_SAMPLES = SAMPLING_RATE*10
	BATCH_SIZE=32
	pickle_dir='featues'
	
	
	mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,n_fft=512,win_length=512,
	hop_length=256,window_fn=torch.hamming_window,n_mels=64 )
	
	freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
	t_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)
		
	if torch.cuda.is_available():
	    device = 'cuda'
	else:
	    device = 'cpu'
	    
	print(f'using {device} device')
	
	#train_asc = asc_dataset(CSV_FILE,BASE_DIR,mel_spectrogram,SAMPLING_RATE,NUM_SAMPLES,device)
	
	print("Creating Training and Validation PyTorch Datasets & Batch Iterators : ===================>\n")
	tf_name = 'train_pickle_aug.pkl'
	vl_name = 'valid_pickle_aug.pkl'
	
	if(os.path.isfile(pickle_dir+'/'+ tf_name)):
		file = open(pickle_dir+'/'+tf_name, 'rb')
		train_asc = pickle.load(file)
		        
	else:
	    train_asc = asc_dataset(CSV_FILE,BASE_DIR,[mel_spectrogram,freq_mask,t_mask],SAMPLING_RATE,NUM_SAMPLES,device)
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
	
	print(train_iter)
	feat,lab=next(iter(train_iter))
	print(f'train features:{feat}')
	print(f'labels:{lab}')
	
   	
		
   		
   	
