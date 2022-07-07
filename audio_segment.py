import os
from pydub import AudioSegment

csv='train1.csv'
dir='data/'
def seg_audio(csv_file,out_dir):
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    with open(csv_file,'r')as f:
        l=f.read().split('\n')
        l=l[1:]#removing the firstline
        for i in l:
            if i=='':
            	break
            i=i.split(',')[0]
            file_path=dir+i
            
            filename=file_path.split('/')[-1]
            filename=filename.split('.')[0]
            print(filename)
            myaudio=AudioSegment.from_wav(file_path)
            chunksize=10000
            print("**************************")
            for i in range(0,21,10):
                split_audio=myaudio[i:i+chunksize]
                split_audio.export(out_dir+"/"+filename+"@@"+str(i)+'.wav',format="wav")


seg_audio(csv,'newdata')
