'''
Author: your name
Date: 2021-07-27 16:05:42
LastEditTime: 2021-08-04 09:52:30
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/attacks/evasion/zosgd_sum.py
'''
'''
Author: your name
Date: 2021-07-27 16:05:42
LastEditTime: 2021-07-28 10:12:06
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/attacks/evasion/zosgd_sum.py
'''
'''
Author: your name
Date: 2021-07-22 14:03:50
LastEditTime: 2021-07-27 17:09:47
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/attacks/evasion/zones.py
'''
import numpy as np
import copy
from eta.attacks.evasion.abstract_evasion_sum import AbstractEvasionSum
np.random.seed(10)
class ZOSGDShapSumMethod(AbstractEvasionSum):
    def gradient_estimation_sum(self,mu,q,x,kappa,target_label,const,weights):
        sigma = 100
        grad_est=0
        d=x.shape[1]
        f_0,ignore=self.function_evaluation_cons_sum(x,kappa,target_label,const,x,weights)
        for i in range(q):
            import_index=np.array([36,12,72,14,22])
            len_import=import_index.shape[0]
            u_import = np.random.normal(0, sigma, (1,len_import))
            u_norm_import = np.linalg.norm(u_import)
            u_import = u_import/u_norm_import
            u=np.zeros(d)
            u[import_index]=u_import
            f_tmp, ignore = self.function_evaluation_cons_sum(x+mu*u,kappa,target_label,const,x,weights)
            grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/(mu)
        return grad_est

