import numpy as np 
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.repair.bounds_repair import repair_random_init
from pymoo.core.variable import Real, get
#Operadores de interés 
class Simplex_Repair(Repair):
    def _do(self, problem, X, **kwargs): 
        X[X < 1e-5] = 0
        return X / X.sum(axis=1, keepdims=True)
    
#Sampling 
class Sampling_Energy(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        #100 individuos (a lo más 2000)
        #problem.n_var la cantidad de activos. 
        X = get_reference_directions("energy", problem.n_var, n_samples)
        return X 
    
class Sampling_Uniform(Sampling): 
    def _do(self, problem, n_samples, **kwargs): 
        X = self.n_uniform_sampling_simplex(problem.n_var, n_samples)
        return X
    
    def uniform_sampling_simplex(self, n_vars):
        sum = 0
        x = np.zeros(n_vars)
        for i in range(n_vars-1):
            u = np.random.uniform()
            #print(u)
            x[i] = (1-sum)*(1- np.power(u, 1/(n_vars-1-i)))
            sum= sum+x[i]
        x[-1]= 1-sum
        return x

    def n_uniform_sampling_simplex(self, n_vars, n_samples):
        X = np.empty((n_samples, n_vars))
        for i in range(n_samples):
            X[i] = self.uniform_sampling_simplex(n_vars)
        return X  
    
#Crossovers 
class SPX(Crossover):
    def __init__(self,
                n_offsprings=3,
                n_parents   =3,
                epsilon = 0.001,
                **kwargs):
      super().__init__(n_parents, n_offsprings, **kwargs)
      self.epsilon =  epsilon

    def _do(self, problem, X, **kwargs):
      n_parents, n_matings, n_var = X.shape
      #print(X.shape)
      Y = np.full((self.n_offsprings,  n_matings, n_var), None)
      for k in range(n_matings):
          parents = X[:, k, :]
          O = np.mean(parents, axis=0)
          yks = [O + (1+self.epsilon) * (p - O) for p in parents]  
          for i in range(self.n_offsprings):   
            Y[i, k, :] = self.get_offspring(yks) 
            #print(Y[i,k,:])
            #print(np.sum(Y[i, k, :]))
      return Y 
           
    def get_offspring(self,yks): 
        n = len(yks)
        rs = np.power(np.random.rand(n), 1 / (np.arange(n) + 1))        
        ck = 0  
        for k in range(1, self.n_parents):
            ck = rs[k-1]*(yks[k-1] - yks[k]+ ck) 
        return yks[-1] + ck    

#Mutation   
def mut_dirichlet(X, xl, xu, prob):
    n, n_var = X.shape
    assert len(prob) == n
    Xp = X[:]
    mut = np.random.rand(n) < prob
    if np.sum(mut)>0:
      #print(X[mut])
      Xp[mut] = np.apply_along_axis( lambda alphas: np.random.dirichlet(alphas), 1, X[mut]+0.01)
    Xp = repair_random_init(Xp, X, xl, xu)
    return Xp

class Probability_Mutation(Mutation):
    def __init__(self, distribution_fun,  **kwargs):
        self.distribution_fun = distribution_fun
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        X = X.astype(float)
        prob_var = self.get_prob_var(problem, size=len(X))
        Xp = self.distribution_fun(X, problem.xl, problem.xu, prob_var)
        return Xp