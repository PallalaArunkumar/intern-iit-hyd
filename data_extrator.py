import os
import pandas as pd
import shutil

df1=pd.read_csv("evaluation/evaluate/shopping_mall.csv",header=None)
directory='data/'

des='shopping_mall_2020data/audio/'

#print(os.path.exists(des))

if not os.path.exists(des):
  os.makedirs(des)
  print("created directory")


for i in range(1,len(df1)):
  file_path=os.path.join(directory,df1.iloc[i][0])
  print(f"{file_path}")
  shutil.copy(file_path,des)
