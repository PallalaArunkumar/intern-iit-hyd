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



def evaluate(model,test_loader,csv_file,device):
    correct,total=0,0
    model.eval()
    with torch.no_grad():
        y_actual_val,y_pred_val=[],[]  ##lists to save target and predicted values
        for i,data in enumerate(test_loader):
        	x,y=data
        	x=x.to(device,dtype=torch.float)
        	y=y.to(device,dtype=torch.long)
        	y_hat=model(x)
        	_,predict=torch.max(y_hat,dim=1)
        	predict = predict.detach().cpu().numpy() ## moving from gpu to cpu and tensor --> numpy
        	y = y.detach().cpu().numpy()
        	y_actual_val.append(y) #to convert into 1D hstack array
        	y_pred_val.append(predict)
        	#print(target , predict) #target values and predicted values
        	total += y.shape[0]
        	correct +=(predict==y).sum().item() #if pred==target then count increses
            
            
	                
    print('Accuracy of the network: %d %%' % (100 * correct / total))
    
    y_val = np.hstack(y_actual_val)
    #print(f'targets:{len(y_val)}')
    y_pred_val = np.hstack(y_pred_val)
    #print(f'predictions:{len(y_pred_val)}')
        
    
    conf_matrix = confusion_matrix(y_val,y_pred_val)
    #np.save('conf_mat_2017_test_withaug_5sec_model215',conf_matrix)
    print("\n\nConfusion matrix:")
    print(conf_matrix)
    
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
	

	te_csv='../test2017/evaluation_setup/test_3class_evset.txt'
	
	BASE_DIR='../test2017'
	pickle_dir='featues'
	

	te_name = 'test2017_pickle.pkl'


	if(os.path.isfile(pickle_dir+'/'+ te_name)):
		file = open(pickle_dir+'/'+te_name, 'rb')
		test_asc = pickle.load(file)
		        
	else:
	    test_asc = asc_dataset(te_csv,BASE_DIR,[mel_spectrogram,freq_mask,t_mask],SAMPLING_RATE,NUM_SAMPLES,device)
	    
	    with open(pickle_dir+'/'+te_name,'ab')as test_pickle:
	    	pickle.dump(test_asc,test_pickle)

	print(f'total test files:{len(test_asc)}')
	
	test_iter = load_test_data(BATCH_SIZE,test_asc)
	
	
	model=xvecTDNN(3,0).to(device)
	s_dict = torch.load('model2017_3class_withaug_3sec/Epoch-237.pth')
	print("model2017_3class_withaug_3sec/Epoch-237.pth")
	model.load_state_dict(s_dict)
	print(evaluate(model,test_iter,te_csv,device))
	print("#################################################################")

    
	
##################################### checking ##############################################################
'''
    for d in device_list:
    	print(f'device:{d}')
    	d_c,c_t=0,0
    	for k in range(len(device_idxs)):
    		if (device_idxs[k]==d):
    			print(k)
    			d_c+=1
    			print(f'y_preds{y_pred_val[k]}')
    			print(f'y_actual:{y_val[k]}')
    			if(y_pred_val[k]==y_val[k]):
    				c_t+=1
    	print(f'device count:{d_c} and correctly preds count:{c_t}')
    	d_a=c_t/d_c
    	print('device accuracy:{d_a*100}')'''
    	
#######################################################################################################	
'''    
#################################################################################
################### Device wise accuracy ########################################

    dev_test_df = pd.read_csv(csv_file,sep=' ',header=None)
    wav_paths = dev_test_df[0].tolist()
    ClassNames = np.unique(dev_test_df[1])
	
    
    for idx, elem in enumerate(wav_paths):
    	wav_paths[idx] = wav_paths[idx].split('/')[-1].split('.')[0]
    	wav_paths[idx] = wav_paths[idx].split('-')[-1]
    	

	
	
    device_idxs = wav_paths
    device_list = np.unique(device_idxs) 	

    device_acc = []
	
    for device_id in device_list:
    	cur_preds = np.array([y_pred_val[i] for i in range(len(device_idxs)) if device_idxs[i] == device_id])
    	cur_y_val = [y_val[i] for i in range(len(device_idxs)) if device_idxs[i] == device_id] #actual target
    
    	cur_acc = np.sum(cur_y_val==cur_preds) / len(cur_preds)
    
    	device_acc.append(cur_acc)
    	
    #print(len(cur_preds))
    #print(cur_preds)
    #print(len(cur_y_val))
    #print(cur_y_val)	
    
    print("\n\nDevices list: ", device_list)
    print("Per-device val acc : ", np.array(device_acc))
    print("Device A acc: ", "{0:.3f}".format(device_acc[0]))
    print("Device B & C acc: ", "{0:.3f}".format((device_acc[1] + device_acc[2]) / 2))
    print("Device s1 & s2 & s3 acc: ", "{0:.3f}".format((device_acc[3] + device_acc[4] + device_acc[5]) / 3))
    print("Device s4 & s5 & s6 acc: ", "{0:.3f}".format((device_acc[6] + device_acc[7] + device_acc[8]) / 3))'''
