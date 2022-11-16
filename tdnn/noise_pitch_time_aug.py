import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle5 as pickle
import soundfile as sf

file_path = ''
csv_file = 'evaluation_setup/uni/train_3class.txt'
output_path = 'audio/'


sr = 44100

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep=' ', header=None)
wavpath = data_df[0].tolist()
labels = data_df[1].tolist()

with open('train_aug_3class.txt','w+')as f:
	for i in range(len(wavpath)):
		stereo, fs = librosa.load(file_path + wavpath[i],sr=None)
		file_name = wavpath[i].split('/')[1].split('.')[0] # getting the filename
		f.write(wavpath[i]+" "+labels[i]+'\n')
		print(wavpath[i]+" "+labels[i]+'\n')
		
		########### Noise #####################
		noise = np.random.normal(0,1,len(stereo))
		augmented_data = np.where(stereo != 0.0, stereo.astype('float64') + 0.01 * noise, 0.0).astype(np.float32)
		sf.write(output_path+file_name+'_noise.wav',augmented_data,sr)
		f.write(output_path+file_name+'_noise.wav'+" "+labels[i]+'\n')
		print(output_path+file_name+'_noise.wav'+" "+labels[i]+'\n')
		
		############ Pitch #######################
		n_step = np.random.uniform(-4, 4)
		y_pitched = librosa.effects.pitch_shift(stereo, sr=44100, n_steps=n_step)
		sf.write(output_path+file_name+'_pitch.wav',y_pitched,sr)
		f.write(output_path+file_name+'_pitch.wav'+" "+labels[i]+'\n')
		print(output_path+file_name+'_pitch.wav'+" "+labels[i]+'\n')
		
		############### time #######################
		time_factor = np.random.uniform(0.5, 2)
		length = len(stereo)
		y_stretch = librosa.effects.time_stretch(stereo, rate=time_factor)
		if len(y_stretch) < length:
		    y_stretch = np.concatenate((y_stretch, y_stretch))
		    y_stretch = y_stretch[0:length]
		else:
		    y_stretch = y_stretch[0:length]
		    
		sf.write(output_path+file_name+'_time.wav',y_stretch,sr)
		f.write(output_path+file_name+'_time.wav'+" "+labels[i]+'\n')
		print(output_path+file_name+'_time.wav'+" "+labels[i]+'\n')
		

        
        

