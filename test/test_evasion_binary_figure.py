'''
Author: your name
Date: 2021-07-20 11:37:47
LastEditTime: 2021-08-02 12:32:28
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_evasion_binary.py
'''
'''
Author: your name
Date: 2021-04-01 15:20:27
LastEditTime: 2021-07-27 16:54:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_data.py
'''
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
import numpy as np
from eta.get_model_data import get_datasets, models_train,parse_arguments,models_train,print_results,models_predict,attack_models_train
from eta.attacks.evasion.evasion_attack import EvasionAttack
import torch
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
import random
from math import ceil
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
# setup_seed(20)

def plot_headmap(X,a_score,model_name):
    X_x=X[:,0]
    X_y=X[:,1]
    plt.scatter(X_x, X_y, marker='o', c=a_score, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(model_name)
    plt.show()

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm
    X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
    X_train,y_train,X_test,y_test=X_train,y_train,X_test[0:500],y_test[0:500]
 
    # X_train_1=X_train[y_train==1]
    # lower=np.min(X_train_1,axis=0)
    # upper=np.max(X_train_1,axis=0)

    lower=0
    upper=1

    setup_seed(20)
    attack_model=attack_models_train(options,False,X_train,y_train)
    setup_seed(20)
    trained_models=models_train(options,False,X_train,y_train)
    y_test,y_pred=models_predict(trained_models,X_test,y_test)

    X_test_1=X_test[y_test==1]
    y_test_1=y_test[y_test==1]
    X_test_0=X_test[y_test==0]
    y_test_0=y_test[y_test==0]
    print(X_test_1.shape)
    print(X_test_0.shape)

    # for i in range(len(y_pred)):
    #      y_pred_1=y_pred[i][y_test==1]
    #      plot_headmap(X_test_1,y_pred_1[:,1],orig_models_name[i])

    table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')
    # explainer=shap.KernelExplainer(attack_model.predict,X_test)
    # explainer = shap.TreeExplainer(attack_model.classifier.model)
    # shap_values = explainer.shap_values(X_test)
    feature_importance=None
    # feature_importance=np.abs(shap_values).mean(0)
    # print(feature_importance)
    X_adv,X_adv_path,y_adv,objective_function=EvasionAttack(attack_model,feature_importance,evasion_algorithm,X_test_1,y_test_1,upper,lower,mask)
    # for i in range(0,10):
    #     dis=X_adv[i]-X_test_1[i]
    #     print(np.linalg.norm(dis,ord=2))
    X_test_adv = np.append(X_test_0, X_adv, axis=0)
    y_test_adv = np.append(y_test_0, y_adv, axis=0)

    y_test_adv,y_adv_pred=models_predict(trained_models,X_test_adv,y_test_adv)

    print_results(datasets_name,orig_models_name,y_test_adv,y_adv_pred,'adversarial_accuracy')

