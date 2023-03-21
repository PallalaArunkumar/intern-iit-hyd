import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
import librosa
import torch.optim as optim
import torch.nn.functional as F
from train_pytorch import *
from model_torch import *
from asc_class import *

def test(model,audio_clip,mapping,device):
	model.eval()
	audio_clip=audio_clip.to(device,dtype=torch.float32)
	output=model(audio_clip)
	_,predict = torch.max(output,dim=1)
	pred = predict.cpu().numpy()
	#print(pred[0],type(pred))
	
	return mapping[pred[0]]

if __name__ == "__main__":
	SAMPLING_RATE = 16000
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
	
	print("model2017_3class_withaug_3sec/Epoch-237.pth")

	audio_clip = os.listdir('zoom_mic/bus_stop/')
	for clip in audio_clip:
		print("Actual:",clip)
		wav,fs=torchaudio.load('zoom_mic/bus_stop/'+clip)
		t1=torchaudio.transforms.Resample(fs,SAMPLING_RATE)
		wav = t1(wav)
		if wav.shape[0]>1:
			wav = torch.mean(wav, dim=0, keepdim=True)
			
			
		#print("number of channels",wav.shape[0],"sampling rate",wav.shape[1])	
		mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,n_fft=512,win_length=512,
			hop_length=256,window_fn=torch.hamming_window,n_mels=64)
			
		signal = mel_spectrogram(wav)
			#print(signal.shape, type(signal))
		signal = signal.log()
			#print(wav.shape)
			
		classes={0:'Indoor',1:'outdoor',2:'transport'}
					 
		model=xvecTDNN(3,0).to(device)
		s_dict = torch.load('model2017_3class_withaug_3sec/Epoch-237.pth')
		
		model.load_state_dict(s_dict)
		print("predicted          :",test(model,signal,classes,device))
