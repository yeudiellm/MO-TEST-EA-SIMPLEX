import pandas as pd 
import numpy as np 
from tqdm import tqdm
#(n_ejec, n_gen, variable_size, n_var )
def get_best_opt(population, population_var,A = None, A_vars=None,  tol=1e-8):
  pop_size, n_obj = population.shape
  _, n_var = population_var.shape
  #Copia de poblacion
  population = population[:]
  #Guardamos los índices 
  indx = range(pop_size)
  #Archivo fantasma inicial 
  if A is None: 
      A = np.array( [[np.inf]*n_obj])
      A_vars = np.array([[np.inf]*n_obj])
  #best_idx = [None]
  #Iterar sobre los portafolios
  for row in population:
  #for idx, row in zip(indx, population):
    test1 = (A <= row).all(axis=1)
    test2 = np.linalg.norm(A-row, ord=2, axis=1) > tol
    if not ((test1) & (test2)).any(): 
      A = np.vstack([A,row])
      A_vars = np.vstack([A_vars, row])
      #best_idx.append(idx)
      test1 = (row <= A).all(axis=1)
      test2 = np.linalg.norm(row- A, ord=2, axis=1)> tol
      A = A[~((test1) & (test2)) ,:]
      A_vars = A_vars[~((test1) & (test2)) ,:]
      #best_idx = list(compress(best_idx,~((test1) & (test2))))
  return A, A_vars

