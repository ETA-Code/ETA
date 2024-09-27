'''
Author: your name
Date: 2021-03-24 19:36:20
LastEditTime: 2021-07-10 19:37:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/classic/random_forest.py
'''
from eta.classifiers.abstract_model import AbstractModel
from sklearn.ensemble import RandomForestClassifier
from eta.estimators.classification.scikitlearn import SklearnClassifier

class RandomForest(AbstractModel):

    def __init__(self,
            n_trees=10,
            split_criterion='gini',
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=0
        ):
        model = RandomForestClassifier(
            n_estimators=n_trees,
            criterion=split_criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.classifier = SklearnClassifier(model=model)
