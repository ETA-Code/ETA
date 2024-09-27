import sys

from pandas.core import base
sys.path.append("")
import matplotlib
matplotlib.use('TKAgg')
import eta.Interpretability.fai as fai
import os

import matplotlib.pyplot as plt
import pandas as pd
from eta.get_model_data import get_datasets,get_datasets_reg,parse_arguments,models_train,print_results,models_predict,attack_models_train
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import log_loss
from itertools import combinations
# from xgboost import plot_importance
torch.set_default_tensor_type(torch.DoubleTensor)

def plot_feature_importance_all(importance, ax=None, height=0.5,
                    xlim=None, ylim=None, title='Feature importance',
                    xlabel='F score', ylabel='Features', max_num_features=None,
                    grid=True, show_values=True, **kwargs):

    if not importance:
        raise ValueError('Booster.get_score() results in empty')

    tuples = [(k, importance[k]) for k in importance]
    if max_num_features is not None:
        # pylint: disable=invalid-unary-operand-type
        tuples = sorted(tuples, key=lambda x: x[1])[-max_num_features:]
    else:
        tuples = sorted(tuples, key=lambda x: x[1])
    labels, values = zip(*tuples)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    if show_values is True:
        for x, y in zip(values, ylocs):
            ax.text(x, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if not isinstance(xlim, tuple) or len(xlim) != 2:
            raise ValueError('xlim must be a tuple of 2 elements')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        if not isinstance(ylim, tuple) or len(ylim) != 2:
            raise ValueError('ylim must be a tuple of 2 elements')
    else:
        ylim = (-1, len(values))
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    # plt.show()
    plt.savefig('experiments/important/'+datasets_name+'/'+options.attack_models+'/'+title+'.pdf', format='pdf',bbox_inches='tight',dpi=1000,transparent=True)
    return ax

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm

    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)

    attack_model=attack_models_train(options,False,X_train,y_train)
    trained_models=models_train(options,False,X_train,y_train)

    y_test,y_pred=models_predict(trained_models,X_test,y_test)
    print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

    imputer = fai.MarginalImputer(attack_model,X_test[:100])
    estimator = fai.PermutationEstimator(imputer, 'cross entropy')
    # estimator = fai.IteratedEstimator(imputer, 'cross entropy')
    fai_values = estimator(X_test, y_test)

    fai_values.plot(max_features=20)
    # print(fai_values.values)
    # plt.show()

    path='experiments/important/'+datasets_name+'/'+options.attack_models
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
    plt.savefig(path+'/ddos_important.pdf', format='pdf',bbox_inches='tight',dpi=1000,transparent=True)

    # argsort = np.argsort(fai_values.values)[::-1]
    # group_cor=list(combinations(argsort[:5], 2))
    # group_single=list(combinations(argsort[:5], 1))
    # groups=group_cor+group_single
    # group_names = [str(i).replace('(','').replace(')','').replace(',','') for i in groups]

    # imputer = fai.GroupedMarginalImputer(attack_model,X_test[:100],groups)
    # # estimator = fai.PermutationEstimator(imputer, 'cross entropy')
    # estimator = fai.IteratedEstimator(imputer, 'cross entropy')
    # # estimator = fai.KernelEstimator(imputer, 'cross entropy')
    # fai_values = estimator(X_test, y_test)
    # # print(fai_values.values)

    # dic_fai = dict(zip(group_names,fai_values.values))
    # dic_cor=dict(zip(group_names[:len(group_cor)],fai_values.values[:len(group_cor)]))
    # for key in dic_cor.keys():
    #     dic_cor[key]=dic_fai[key]-(dic_fai[key.split(' ')[0]]+dic_fai[key.split(' ')[1]])
    # print(dic_fai)
    # print(dic_cor)

    # plot_feature_importance_all(dic_fai,title='Feature importance')
    # plot_feature_importance_all(dic_cor,title='Cor importance',show_values=False)
            
    # fai_values.plot(group_names)
    # plt.show()


