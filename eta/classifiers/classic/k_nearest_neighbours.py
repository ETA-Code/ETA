'''
Author: your name
Date: 2021-03-24 19:34:34
LastEditTime: 2021-07-10 19:27:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/classic/k_nearest_neighbours.py
'''
from sklearn import model_selection
from eta.classifiers.abstract_model import AbstractModel
from sklearn.neighbors import KNeighborsClassifier
from eta.estimators.classification.scikitlearn import SklearnClassifier

class KNearestNeighbours(AbstractModel):
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        model= KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm
        )
        self.classifier = SklearnClassifier(model=model)