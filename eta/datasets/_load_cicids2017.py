'''
Author: your name
Date: 2021-03-25 14:30:40
LastEditTime: 2021-07-18 13:32:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/datasets/_loadcic.py
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
from os import path
from eta.datasets._base import get_mask,get_true_mask,add_str,get_dict,train_val_test_split

def load_cicids2017():
    file_path='eta/datasets/data/CIC-IDS-2017/botnet.pkl'
    df = pd.read_pickle(file_path)

    df.columns = df.columns.str.lstrip()
    df['Flow Bytes/s'] = pd.to_numeric(df['Flow Bytes/s'], errors='coerce')
    df['Flow Packets/s'] = pd.to_numeric(df['Flow Packets/s'], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)

    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)

    # print(df)

    df_1=df[df['Label']==1]
    df_0=df[df['Label']==0]

    #botnet
    df_0 = df_0.sample(frac=0.02, random_state=20)
    df_1 = df_1.sample(frac=0.99, random_state=20)

    #ddos
    # df_0 = df_0.sample(frac=0.005, random_state=20)
    # df_1 = df_1.sample(frac=0.005, random_state=20)

    #brute_force
    # df_0 = df_0.sample(frac=0.02, random_state=20)
    # df_1 = df_1.sample(frac=0.5, random_state=20)

    # web_attack
    # df_0 = df_0.sample(frac=0.02, random_state=20)
    # df_1 = df_1.sample(frac=0.99, random_state=20)

    print('0-'+str(df_0.shape[0]))
    print('1-'+str(df_1.shape[0]))
    
    df= pd.concat([df_0,df_1])

    # writer=pd.ExcelWriter('my.xlsx')
    # df.to_excel(writer,float_format='%.5f')
    # writer.save()

    df['Label']=df['Label'].astype("int")
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]
    # X = df.drop(['Label'], axis=1)
    # y = df['Label']
    # print(X.shape)
    mask=get_true_mask([column for column in X])
    return X,y,mask
