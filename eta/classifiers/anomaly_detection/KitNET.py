'''
Author: your name
Date: 2021-07-15 14:48:32
LastEditTime: 2021-08-05 10:01:01
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/anomaly_detection/KitNET.py
'''
'''
Author: your name
Date: 2021-07-15 14:48:32
LastEditTime: 2021-07-21 20:12:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/anomaly_detection/KitNET.py
'''
from eta.classifiers.abstract_model import AbstractModel
from sklearn import ensemble
import numpy as np
from eta.classifiers.anomaly_detection.KitNET_packet.model import tKitsune,eKitsune
from eta.classifiers.anomaly_detection.KitNET_packet.model import RunKN
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler

class KitNET(AbstractModel):

    def __init__(self,input_size,output_size):
        self.feature_size=input_size
        self.AD_threshold=0
        self.model_save=[]

    def train(self, X_train, y_train):
        X_train=X_train[y_train==0]
        fm_n=int(len(X_train)/10)
        ad_n=len(X_train)-fm_n
        model=tKitsune(self.model_save, self.feature_size, 10, fm_n, ad_n)
        rmse = RunKN(model, X_train)
        self.AD_threshold = max(rmse[fm_n:])
        self.model_save.append(self.AD_threshold)

    def predict(self, X):
        model = eKitsune(self.model_save, self.feature_size, 10)
        rmse = RunKN(model, X)
        rmse = np.array(rmse)
        # kitsune-botnet
        AD_threshold = self.model_save[3]
        # kitsune-ddos
        # AD_threshold = 0.015
        # cicids2017-botnet
        # AD_threshold=0.016
        # cicids2017-ddos
        # AD_threshold=0.01
        # rmse[rmse > AD_threshold]=1
        # rmse[rmse <=AD_threshold]=0
        anomaly_scores=rmse.reshape(-1,1)
        y_pred=np.hstack((-(anomaly_scores-AD_threshold),anomaly_scores-AD_threshold))
        # print(y_pred)
        
        return y_pred

    def predict_label(self, X):
        pred=self.predict(X)
        label=np.argmax(pred, axis=1)
        return label

    def predict_abnormal(self,X):
        pred=self.predict(X)
        # max_pred=np.max(pred,axis=1)
        max_pred=pred[:,1]
        return max_pred
        
    def save(self, path):
        return 0

    def load(self, path):
        return 0
        
        