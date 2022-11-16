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

	audio_clip = os.listdir('sound_box/')
	for clip in audio_clip:
		print("Actual",clip)
		if clip=='test_real':
				continue
		wav,fs=torchaudio.load('sound_box/'+clip)
		t1=torchaudio.transforms.Resample(fs,SAMPLING_RATE)
		wav = t1(wav)
			#print(type(wav))
			
			
		mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,n_fft=512,win_length=512,
			hop_length=256,window_fn=torch.hamming_window,n_mels=64)
			
		signal = mel_spectrogram(wav)
			#print(signal.shape, type(signal))
		signal = signal.log()
			#print(wav.shape)
			
		classes={0:'Indoor',1:'Forest_path',2:'outdoor',3:'transport'}
					 
		model=xvecTDNN(4,0.3).to(device)
		s_dict = torch.load('models_4class/Epoch-198.pth')
		model.load_state_dict(s_dict)
		print(test(model,signal,classes,device))
