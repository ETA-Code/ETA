'''
Author: your name
Date: 2021-03-24 16:09:20
LastEditTime: 2021-07-10 19:53:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/ml_ids/abstract_model.py
'''
from joblib import dump, load
import numpy as np

class AbstractModel:
    """
    Base model that all other models should inherit from.
    Expects that classifier algorithm is initialized during construction.
    """

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_label(self, X):
        pred=self.classifier.predict(X)
        label=np.argmax(pred, axis=1)
        return label

    def predict_abnormal(self,X):
        pred=self.classifier.predict(X)
        # max_pred=np.max(pred,axis=1)
        max_pred=pred[:,1]
        return max_pred

    def save(self, path):
        dump(self.classifier, path)

    def load(self, path):
        self.classifier = load(path)
