import numpy as np
import pandas as pd 
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

class Evaluator_Solutions: 
    def __init__(self, Mega_X: np.ndarray, Mega_F: np.ndarray):
        self.n_ejec, self.n_gen, self.n_pop_size, self.n_var = Mega_X.shape
        _, _ , _ , self.n_obj = Mega_F.shape
        #Falta definir las salidas del HV 
        self.Mega_X = Mega_X[:]
        self.Mega_F = Mega_F[:]
        self.Mega_HV= np.empty((self.n_ejec, self.n_gen))
        self.Mega_HV_Opt = np.empty((self.n_ejec, self.n_gen))
            
    def get_opt_HyperVolume(self, ref_point, tol=1e-4):
        temp_Mega_F = np.reshape(self.Mega_F, newshape=(-1, self.n_obj))
        scaler      = MinMaxScaler()
        scaler.fit(temp_Mega_F) 
        temp_Mega_F = None  #Save RAM memory
        ind = HV(ref_point=ref_point)
        for i in tqdm(range(self.n_ejec)): 
            F_opt = None 
            X_opt = None
            for j in range(self.n_gen): 
                self.Mega_HV[i,j] = ind(scaler.transform(self.Mega_F[i,j]))
                #F_opt, X_opt =get_best_opt(self.Mega_F[i,j], self.Mega_X[i,j], F_opt, X_opt, tol)
                #F_opt, A_opt      = get_best_opt(self.Mega_F[i,j]  )
                self.Mega_HV_Opt[i,j] = ind(scaler.transform(F_opt))
        return 
    
    def get_report(self, name_file=''):
        Mega_HV = np.reshape(self.Mega_HV, newshape=(-1,))
        Mega_HV_opt = np.reshape(self.Mega_HV_Opt, newshape=(-1))
        Generacion = list(range(self.n_gen))*self.n_ejec
        Problema   = [self.name_problem]*(self.n_ejec*self.n_gen)
        Algoritmo  = [self.name_algorithm]*(self.n_ejec*self.n_gen)
        Ejecucion = []
        for s in range(self.n_ejec): 
            Ejecucion = Ejecucion+[s+1]*self.n_gen
        result= {'Algoritmo':Algoritmo,
                 'Problema':Problema, 
                 'Ejecucion':Ejecucion, 
                 'Generacion': Generacion, 
                 'HV_gen':Mega_HV, 
                 'HV_opt':Mega_HV_opt}
        result = pd.DataFrame(result)
        result.to_csv(name_file+'.csv', index=False)
        return result   