from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.problems import get_problem

import numpy as np 
import pandas as pd
####################################################################################################
#Pymoo problems modifications
from pymoo.problems.multi import ZDT4
import pymoo.gradient.toolbox as anp
class ZDT4_BIS(ZDT4):
  def __init__(self, n_var=10):
    super().__init__(n_var)
    self.xl = 0 * np.ones(self.n_var)
    self.xl[0] = 0.0
    self.xu = 1 * np.ones(self.n_var)
    self.xu[0] = 1.0
    self.func = self._evaluate

################################################################################################
#Utility Functions for population
def get_full_population(res):     
    all_pop = Population()
    for algo in res.history:
        all_pop = Population.merge(all_pop, algo.off)
    X = all_pop.get('X')
    F = all_pop.get('F')
    return X, F 
#Utility Functions for problem
def get_problem_bis(name, *args, **kwargs):
    name = name.lower()
    if name.startswith("zdt"): 
        if name=="zdt4": 
            return ZDT4_BIS()
        else: 
            return get_problem(name)
    else: 
        return get_problem(name, *args, **kwargs)
#Utility Functions for algorithm of interest  
def get_algorithm(name, *args, **kwargs): 
    name = name.lower()
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.algorithms.moo.age import AGEMOEA
    ALGORITHM = {'nsgaii': NSGA2, 
                 'nsgaiii': NSGA3,
               'sms_emoa': SMSEMOA, 
               'age_moea': AGEMOEA, 
               'moead': MOEAD
               }
    if name not in ALGORITHM:
        raise Exception("Algorithm not found.")

    return ALGORITHM[name](*args, **kwargs)

#############################################################################################################################
#SOLUTIONS GENERATOR FOR SEVERAL PROBLEMS 
class Generator_Solutions: 
    def __init__(self, n_ejec, n_gen, name_problem, name_algorithm, kwargs_problem, kwargs_algorithm): 
        #EJECUTIONS AND GENERATIONS
        self.n_ejec = n_ejec 
        self.n_gen  = n_gen 
        #PYMOO PROBLEM 
        self.name_problem = name_problem
        self.problem = get_problem_bis(name_problem, **kwargs_problem)
        #PYMOO ALGORITHM 
        self.name_algorithm = name_algorithm 
        self.algorithm = get_algorithm(name_algorithm, **kwargs_algorithm)
        #PYMOO VARIABLES 
        self.n_obj = self.problem.n_obj
        self.n_var = self.problem.n_var
        self.n_pop_size = self.algorithm.pop_size
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

#SEARCHING FAKE
def get_solutions_fake(problems, vars, objs, n_ejec, n_gen, name_algorithm, kwargs_algorithm, name_oprt=''): 
    for i, name_problem in enumerate(problems): 
        print(name_problem)
        sols = Generator_Solutions(n_ejec, n_gen, name_problem, name_algorithm, 
                                   {'n_var': vars[i], 'n_obj': objs[i]}, 
                                   kwargs_algorithm)
        sols.get_extensive_population(save=False, name_files='Solutions/'+name_algorithm+'/'+name_problem+'_'+name_oprt)
    return