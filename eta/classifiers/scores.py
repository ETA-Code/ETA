'''
Author: your name
Date: 2021-04-01 16:16:26
LastEditTime: 2021-08-03 17:16:20
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/scores.py
'''
'''
Author: your name
Date: 2021-04-01 16:16:26
LastEditTime: 2021-07-27 10:45:30
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/scores.py
'''
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tabulate import tabulate

def get_class_scores(labels, predictions):
    accuracy = metrics.accuracy_score(labels, predictions)
    if len(set(labels.flatten().tolist()))<=2:
        f1 = metrics.f1_score(labels, predictions)
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)
        fpr,tpr,trhresholds = roc_curve(y_true=labels,y_score=predictions)
        roc_auc = auc(x=fpr,y=tpr)
        return accuracy, f1, precision, recall,roc_auc
    else:
        f1 = metrics.f1_score(labels, predictions,average='micro')
        precision = metrics.precision_score(labels, predictions,average='micro')
        recall = metrics.recall_score(labels, predictions,average='micro')
        roc_auc=0
        return accuracy, f1, precision, recall,roc_auc
    

def print_results(labels, predictions):
    headers = ['accuracy', 'f1', 'precision', 'recall', 'detection_rate']
    rows=[]
    result=get_class_scores(labels, predictions)
    row=list(result)
    rows.append(row)
    print(tabulate(rows, headers=headers))
