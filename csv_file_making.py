import os
import csv


csv_indir='train1.csv'#reference file
#dir='data/'
def make_csvfile(csv_file):
    
    with open(csv_file,'r')as f:
        l=f.read().split('\n')
        l=l[1:]#removing the firstline
        with open('train1_middle.csv','w')as f:
            for i in l:
                if i=='':
                    break
                i=i.split(',')
                j=i[0]#filename(which is in the first column of csv file)
                k=i[1]#label
                filename=j.split('/')[-1]
                filename=j.split('.')[0]
                for i in range(0,21,10):
                    f.write(filename+"@@"+str(i)+'.wav'+','+k+'\n')#filename and corresonding label


make_csvfile(csv_indir)
