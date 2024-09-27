'''
Author: your name
Date: 2021-04-19 10:22:31
LastEditTime: 2021-07-10 20:17:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/classic/deepforest.py
'''
'''
Author: your name
Date: 2021-04-06 13:57:54
LastEditTime: 2021-04-16 14:09:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/classic/xgboost.py
'''
from eta.classifiers.abstract_model import AbstractModel
from deepforest import CascadeForestClassifier
from eta.estimators.classification.ensemble_tree import EnsembleTree

class DeepForest(AbstractModel):
    def __init__(self,input_size,output_size):
        model=CascadeForestClassifier(random_state=1,verbose=0)
        self.classifier=EnsembleTree(model=model,nb_features=input_size, nb_classes=output_size)
