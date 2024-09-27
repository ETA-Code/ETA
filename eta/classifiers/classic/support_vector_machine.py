'''
Author: your name
Date: 2021-03-24 19:36:59
LastEditTime: 2021-07-10 19:31:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/classic/support_vector_machine.py
'''
from eta.classifiers.abstract_model import AbstractModel
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from eta.estimators.classification.scikitlearn import SklearnClassifier
class SupportVectorMachine(AbstractModel):

    def __init__(self):
        # model =SVC(kernel='linear')
        model=LinearSVC()
        self.classifier = SklearnClassifier(model=model)
    


