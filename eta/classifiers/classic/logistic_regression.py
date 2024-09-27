'''
Author: your name
Date: 2021-03-24 19:35:26
LastEditTime: 2021-07-13 15:23:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/classic/logistic_regression.py
'''
from eta.classifiers.abstract_model import AbstractModel
from sklearn import linear_model
from eta.estimators.classification.scikitlearn import SklearnClassifier

class LogisticRegression(AbstractModel):

    def __init__(self, max_iter=1000, solver='lbfgs'):
        model= linear_model.LogisticRegression(
        multi_class='ovr',
        solver=solver,
        max_iter=max_iter
    )
        self.classifier = SklearnClassifier(model=model)
