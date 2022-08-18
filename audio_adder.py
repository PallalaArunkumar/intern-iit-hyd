import numpy as np
import pandas as pd
from pydub import AudioSegment as AS
import os

c_files=os.listdir('c_data')  # loading all the csv files in the folder
#print(c_files)
#print(type(c_files))

for w in c_files:      #for each csv file we repeat this loop
    q=w[:-4]  			#custom modifications
    class_name=w[:-10]
    label=w[:-10]
    #print(q)
    #print(class_name)
    des=q+'_datafiles'

    if not os.path.exists(des):
        os.makedirs(des)
        print("created directory")
  

    df=pd.read_csv('c_data/'+w,sep=',')#loading the files from the csv files
    class_name=df[(df['scene_label']==label)]
    f1=class_name['filename']

    l1=len(f1)
    for i in range(1,4):
        if (l1%3==0):
            break
        else:
            l1-=1

    print(l1) # checking len coz 3 files --> 1 files i.e why we check the len and then procedue further

    with open(label+'new_30sec.csv','a') as f:     
        for i in range(0,l1,3):
            s1=AS.from_wav(f1[i])
            s2=AS.from_wav(f1[i+1])
            s3=AS.from_wav(f1[i+2])
            c_s=s1+s2+s3
            c_s.export(des+'/'+f1[i+2][6:],format='wav') #saving the audio files.
            print("***",i,"***",i+1,"***",i+2)#just to check
            f.write(des+'/'+f1[i+2][6:]+','+label+'\n') # creating new csv file as data created
        print(label)
