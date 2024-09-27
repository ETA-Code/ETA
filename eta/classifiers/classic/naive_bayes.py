'''
Author: your name
Date: 2021-03-24 19:35:55
LastEditTime: 2021-07-10 19:29:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/classic/naive_bayes.py
'''
from eta.classifiers.abstract_model import AbstractModel
from sklearn.naive_bayes import BernoulliNB
from eta.estimators.classification.scikitlearn import SklearnClassifier

class NaiveBayes(AbstractModel):

    def __init__(self):
        model = BernoulliNB()
        self.classifier = SklearnClassifier(model=model)
