import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm
import pickle5 as pickle
import pytorch_lightning as pl
import torch.multiprocessing as mp
import librosa

class ASC_March(Dataset):
	def __init__(self,base_dir,csv,TARGET_SAMPLING_RATE,device,data_aug=False):
		self.df = pd.read_csv(csv,sep=' ',header=None)
		self.base_dir = base_dir
		self.device = device
		self.target_sampling_rate = TARGET_SAMPLING_RATE
		self.data_aug = data_aug
		
	
	def __len__(self):
		return (len(self.df))
		
	def __getitem__(self,index):
		classes_dict = {'airport': 0, 'bus': 1, 'metro': 2, 'metro_station': 3, 'park': 4, 'public_square': 5, 'shopping_mall': 6, 'street_pedestrian': 7, 'street_traffic': 8, 'tram': 9}
		
		row = self.df.iloc[index]
		#print(row)
		file_path = os.path.join(self.base_dir,row[1])
		#print(file_path)
		signal,sr = torchaudio.load(file_path)
		#signal = signal.to(self.device)
		
		signal = self._resample_if_necessary(signal,sr) ## Resampling the signal
		#print(signal.dtype)
		#print(signal)
		
		if self.data_aug:
			random_choice = np.random.randint(low=0,high=4)
			#print("random choice",random_choice)
			if random_choice == 0:
				return signal,classes_dict[row[0]]
		
			elif random_choice == 1:
				noise_signal = torch.randn((1,self.target_sampling_rate*10))
				scaled_noise = self._ScaledNoise(signal.numpy(),noise_signal.numpy(),3) # adding noise at 3db
				noisy_signal = torch.from_numpy(scaled_noise)+signal
				
				#torchaudio.save('noise_signal.wav',noisy_signal,self.target_sampling_rate)
				return noisy_signal,classes_dict[row[0]]
			
			elif random_choice == 2:
				pitch_factor = np.random.randint(low=-2,high=3)
				
				pitch_signal = librosa.effects.pitch_shift(signal.numpy(),sr=self.target_sampling_rate,n_steps=pitch_factor)
				
				pitch_signal = torch.from_numpy(pitch_signal)
				pitch_signal = self._cut_if_necessary(pitch_signal)
				pitch_signal = self._right_pad_if_necessary(pitch_signal)
				
				#torchaudio.save('pitch_signal.wav',pitch_signal,self.target_sampling_rate)
				return pitch_signal,classes_dict[row[0]]
			
			elif random_choice == 3:
				time_factor = np.random.uniform(0.9,1.3)
				
				speed_signal = librosa.effects.time_stretch(signal.numpy(),rate = time_factor)
				speed_signal = torch.from_numpy(speed_signal)
				speed_signal = self._cut_if_necessary(speed_signal)
				speed_signal = self._right_pad_if_necessary(speed_signal)
				
				#torchaudio.save('speed_signal.wav',speed_signal,self.target_sampling_rate)
				
				return speed_signal,classes_dict[row[0]]
		else:
			return signal,classes_dict[row[0]]

	def _resample_if_necessary(self,signal,sr):
		if( sr != self.target_sampling_rate):
			resampler = torchaudio.transforms.Resample(sr,self.target_sampling_rate)
			signal = resampler(signal)
		return signal
		
	#scaled noise
	def _ScaledNoise(self,sig,noise_sig,target_snr):
		sig_power = np.mean(pow(sig,2))
		noise_power = np.mean(pow(noise_sig,2))
		alpha = np.sqrt(sig_power/(noise_power*pow(10,(target_snr/10.0))))
		#print(alpha,(alpha*noise_sig).shape)
		return alpha*noise_sig
		
	def _cut_if_necessary(self,signal):
		if signal.shape[1] > (self.target_sampling_rate*10):
			signal = signal[:,:self.target_sampling_rate*10]
		return signal
		
	def _right_pad_if_necessary(self,signal):
		target_signal_length = self.target_sampling_rate*10
		length_signal = signal.shape[1]
		if length_signal < target_signal_length:
			#print(signal.dtype)
			#print(signal)
			signal = signal.float()
			signal = torch.nn.functional.pad(signal,(0,target_signal_length-length_signal),mode='constant')	
		return signal	

						
			
		
if __name__ == '__main__':
	mp.set_start_method('spawn')
	TARGET_SAMPLING_RATE = 16000
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	base_dir = '../data2020'
	train_csv = '../data2020/evalution_setup_march/train_label_clip.txt'
	valid_csv = '../data2020/evalution_setup_march/test_label_clip.txt'
	
	train_asc = ASC_March(base_dir,train_csv,TARGET_SAMPLING_RATE,device,data_aug=True)
	valid_asc = ASC_March(base_dir,valid_csv,TARGET_SAMPLING_RATE,device,data_aug=False) 
	#print(len(valid_asc))
	
	train_itr = DataLoader(train_asc,256,shuffle=True,num_workers=15)
	test_itr = DataLoader(valid_asc,256,shuffle=False,num_workers=15)
	
	t_f,t_l = next(iter(train_itr))
	print('feature shape',t_f.shape)
	print('corresponding label',t_l)
	
	te_f,te_l = next(iter(test_itr))
	print('feature shape',te_f.shape)
	print('corresponding label',te_l.shape)
	

