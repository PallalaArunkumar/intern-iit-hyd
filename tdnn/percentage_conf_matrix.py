import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

c_matrix=np.load('conf_2016_test_org_meta.npy') #numy file
cmn=c_matrix.astype('float')/c_matrix.sum(axis=1)[:,np.newaxis]
fig,ax =plt.subplots(figsize=(8,8))
lab=['forest_path','indoor','outdoor','transport'] # labels in order
sns.heatmap(cmn,annot=True,fmt='.2%',xticklabels=lab,yticklabels=lab,cmap='Blues')
# plt.xlabel('predicted')
# plt.ylabel('Actual')
plt.savefig("2016_test_4class.png",facecolor='w')
