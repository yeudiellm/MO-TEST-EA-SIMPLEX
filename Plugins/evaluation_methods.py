import numpy as np 
import pandas as pd 
from pymoo.indicators.hv import HV

class Evaluator_Solutions(): 
    def __init__(self,name_algorithm, name_problem, operators, n_ejec, n_obj):
        self.name_algorithm = name_algorithm
        self.name_problem = name_problem 
        self.operators = operators 
        self.n_ejec = n_ejec 
        self.n_obj  = n_obj 
        self.ref_point  = np.array([1.1]*n_obj)
        
    def get_max_min(self): 
        #20 ejecuciones, 5 operadores, 100 ultimas generaciones 
        max_matrix     = np.empty(shape=(self.n_ejec*len(self.operators), self.n_obj))
        min_matrix     = np.empty(shape=(self.n_ejec*len(self.operators), self.n_obj))
        for i,name_oprt in enumerate(self.operators): 
            Mega_F = np.load('Solutions/'+self.name_algorithm+'/'+self.name_problem+'_'+name_oprt+'_F.npy')
            for j in range(self.n_ejec): 
                F = Mega_F[j, -1]
                max_matrix[i*self.n_ejec+j,:] = np.max(F, axis=0)
                min_matrix[i*self.n_ejec+j,:] = np.min(F, axis=0)
        general_max = np.max(max_matrix, axis=0)
        general_min = np.min(min_matrix, axis=0)
        return general_max, general_min
    
    def minmaxScaler(self,X, general_max, general_min, new_max, new_min): 
        X_std = (X -general_min)/(general_max-general_min)
        X_scaled = X_std*(new_max-new_min) + new_min
        return X_scaled
    
    def get_opt_HyperVolume(self):
        self.values_HV  = np.empty(shape=(len(self.operators), self.n_ejec))
        general_max, general_min = self.get_max_min()
        ind = HV(ref_point=self.ref_point)
        for i, name_oprt in enumerate(self.operators): 
            Mega_F = np.load('Solutions/'+self.name_algorithm+'/'+self.name_problem+'_'+name_oprt+'_F.npy')
            for j in range(self.n_ejec): 
                F_norm = self.minmaxScaler(Mega_F[j, -1], general_max, general_min, 1, 0)
                self.values_HV[i,j] = ind(F_norm)
        return 
    
    def get_report(self, save_file=False):
        S = pd.DataFrame()
        for i, name_oprt in enumerate(self.operators):
            result = {'Algorithm': self.name_algorithm, 
                      'Problem' : self.name_problem, 
                      'Operator': name_oprt, 
                      'Execucion': range(self.n_ejec), 
                      'Generation': 249, 
                      'HV_gen': self.values_HV[i]} 
            result = pd.DataFrame(result)
            S = pd.concat([S, result], ignore_index=True)   
        if save_file:
            S.to_csv('Hypervolumes/'+self.name_algorithm+'/'+self.name_problem+'.csv', index=False) 
        return S 
    
def get_final_reports(algorithms, problems, n_obj, operators, n_ejec): 
    for name_algorithm in algorithms: 
        for name_problem in problems: 
            evals = Evaluator_Solutions(name_algorithm, name_problem, operators, n_ejec, n_obj)
            evals.get_opt_HyperVolume()
            final_report = evals.get_report(save_file=True)
    return final_report

def merge_dataframes(name_algorithm, problems): 
    S = pd.DataFrame()
    for name_problem in problems: 
        rr = pd.read_csv('Hypervolumes/'+name_algorithm+'/'+name_problem+'.csv') 
        S  = pd.concat([S, rr])
    return S 

def conteo_winners(name_algorithm, problems, operators):
    dict_winners = {oprt: [0]*len(operators) for oprt in operators} 
    for name_problem in problems: 
        rr = pd.read_csv('Hypervolumes/'+name_algorithm+'/'+name_problem+'.csv') 
        rr = rr.groupby('Operator', as_index=False).agg({'HV_gen': ['mean', 'std']})
        rr.columns = list(map(lambda x: '_'.join(x).rstrip('_'), rr.columns)) 
        rr = rr[rr['Operator'].isin(operators)]
        temp_order  =rr.sort_values(by='HV_gen_mean', ascending=False)['Operator']
        for i, oprt in enumerate(temp_order): 
            dict_winners[oprt][i] +=1
        
    return pd.DataFrame(dict_winners).transpose() 