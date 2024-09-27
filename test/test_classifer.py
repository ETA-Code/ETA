'''
Author: your name
Date: 2021-07-20 15:23:44
LastEditTime: 2021-07-28 09:50:32
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_classifer.py
'''
'''
Author: your name
Date: 2021-04-06 20:18:43
LastEditTime: 2021-07-27 15:59:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_ensemble.py
'''
import sys
sys.path.append("")
from eta.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_and_ensemble_predict
import numpy as np
import torch
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from sklearn.preprocessing import MinMaxScaler
from math import ceil

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    

    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
    X_train,y_train,X_test,y_test=X_train,y_train,X_test,y_test


    X_test_1=X_test[y_test==1]
    X_test_0=X_test[y_test==0]
    print(X_test_1.shape)
    print(X_test_0.shape)

    trained_models=models_train(options,False,X_train,y_train)
    y_test,y_pred=models_predict(trained_models,X_test,y_test)

    table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')









