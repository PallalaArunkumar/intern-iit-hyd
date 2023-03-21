import torch
import torchaudio
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import pickle5 as pickle
from utils import PreEmphasis
from torchsummary import summary

class xvecTDNN(nn.Module): 

	def __init__(self, num_classes, p_dropout):
		super(xvecTDNN, self).__init__()
		
		self.n_mels	 = 64
		self.log_input  = True
		self.instancenorm   = nn.InstanceNorm1d(self.n_mels)
		self.torchfb		= torch.nn.Sequential(
				PreEmphasis(),
				torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512, hop_length=256, window_fn=torch.hamming_window, n_mels=self.n_mels)
				)

		self.tdnn1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, dilation=1)
		self.bn_tdnn1 = nn.BatchNorm1d(64, momentum=0.1, affine=False)
		self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

		self.tdnn2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, dilation=2)
		self.bn_tdnn2 = nn.BatchNorm1d(64, momentum=0.1, affine=False)
		self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

		self.tdnn3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, dilation=3)
		self.bn_tdnn3 = nn.BatchNorm1d(64, momentum=0.1, affine=False)
		self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

		self.tdnn4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, dilation=1)
		self.bn_tdnn4 = nn.BatchNorm1d(64, momentum=0.1, affine=False)
		self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

		self.tdnn5 = nn.Conv1d(in_channels=64, out_channels=200, kernel_size=1, dilation=1)
		self.bn_tdnn5 = nn.BatchNorm1d(200, momentum=0.1, affine=False)
		self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

		self.fc1 = nn.Linear(200*2,512)
		self.fc2 = nn.Linear(512,num_classes)
		
		self.attention = nn.Sequential(
			nn.Conv1d(200,256, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Conv1d(256, 200, kernel_size=1),
			nn.Softmax(dim=2),
			)

	def forward(self,x):
		x  = x.reshape(-1,x.size()[-1])
		Length = x.shape[1]
		with torch.no_grad():
			with torch.cuda.amp.autocast(enabled=False):
				x = self.torchfb(x)+1e-6
				if self.log_input: x = x.log()
				x = self.instancenorm(x).unsqueeze(1)
		
		x = x.squeeze(dim=1)
		#print("after squeezing",x.shape)
		x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
		x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
		x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
		x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
		x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))
		w = self.attention(x)
		#print("self attention layers",w.shape)
		''' 
		#check the attention weights
		#dummy = w
		#dummy = dummy.squeeze().detach().cpu().numpy()
		#print(dummy.shape)
		#print("afterself attention",dummy)
		np.save("public_square_attention_weights.npy",dummy)'''
		mu = x * w
		#print("the values after multiplying with input",mu)
		mu = torch.sum(x * w, dim=2)
		#print("mean dim",mu.shape)
		sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5))
		#print("variance dim",sg.shape)
		stats = torch.cat((mu,sg),1)
		#print("stats shape",stats.shape)
		outp = self.fc1(stats)
		outp = self.fc2(outp)
		return outp
		
		
		
		
if __name__ == "__main__":
	device = "cuda" if torch.cuda.is_available() else "cpu"
	#device = 'cpu'
	model = xvecTDNN(10,0.05)
	summary(model.to(device),(1,160000))
	model = model.to(device)
	
