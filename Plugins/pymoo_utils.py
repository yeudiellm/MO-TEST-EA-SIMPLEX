from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.problems import get_problem

import numpy as np 
import pandas as pd

################################################################################################
#Utility Functions for population
def get_full_population(res):     
    all_pop = Population()
    for algo in res.history:
        all_pop = Population.merge(all_pop, algo.off)
    X = all_pop.get('X')
    F = all_pop.get('F')
    return X, F 
  
def get_algorithm(name, *args, **kwargs): 
    name = name.lower()
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.algorithms.moo.age import AGEMOEA
    ALGORITHM = {'nsgaii': NSGA2, 
               'sms-emoa': SMSEMOA, 
               'age-moea': AGEMOEA, 
               }
    if name not in ALGORITHM:
        raise Exception("Algorithm not found.")

    return ALGORITHM[name](*args, **kwargs)

#############################################################################################################################
#SOLUTIONS GENERATOR FOR SEVERAL PROBLEMS 
class Generator_Solutions: 
    def __init__(self, n_ejec, n_gen, name_problem, name_algorithm, kwargs_problem, kwargs_algorithm): 
        self.n_ejec = n_ejec 
        self.n_gen  = n_gen 
        self.n_obj = kwargs_problem['n_obj']
        self.n_var = kwargs_problem['n_var']
        self.n_pop_size = kwargs_algorithm['pop_size']
        #PYMOO PROBLEM 
        self.name_problem = name_problem
        try: 
            self.problem = get_problem(name_problem, **kwargs_problem)
        except: 
            self.problem = get_problem(name_problem)
        #PYMOO ALGORITHM  
        self.name_algorithm = name_algorithm 
        self.algorithm = get_algorithm(name_algorithm, **kwargs_algorithm)
        #Terminations Criterio 
        self.termination = get_termination("n_gen", n_gen)
        #(n_ejec, n_gen, n_pop_size, n_var)
        #(n_ejec, n_gen, n_pop_size, n_var)
        self.Mega_X = np.empty((self.n_ejec, self.n_gen, self.n_pop_size, self.n_var))
        self.Mega_F = np.empty((self.n_ejec, self.n_gen, self.n_pop_size, self.n_obj))
    
    def get_extensive_population(self, save=True, name_files=''): 
        for i in range(self.n_ejec): 
            res = minimize(problem = self.problem,
                           algorithm = self.algorithm,
                           termination = self.termination, 
                           save_history=True)
            for j, algo in enumerate(res.history): 
                self.Mega_X[i, j] = algo.off.get('X')
                self.Mega_F[i, j] = algo.off.get('F')
        if save: 
            np.save(name_files+'_X.npy', self.Mega_X)
            np.save(name_files+'_F.npy', self.Mega_F)
            
#SEARCHING SOLUTIONS FOR SEVERAL PROBLEMS  
def get_solutions(problems, vars, objs, n_ejec, n_gen, name_algorithm, kwargs_algorithm, name_oprt=''): 
    for i, name_problem in enumerate(problems): 
        print(name_problem)
        sols = Generator_Solutions(n_ejec, n_gen, name_problem, name_algorithm, 
                                   {'n_var': vars[i], 'n_obj': objs[i]}, 
                                   kwargs_algorithm)
        sols.get_extensive_population(save=True, name_files='Solutions/'+name_algorithm+'/'+name_problem+'_'+name_oprt)
    return