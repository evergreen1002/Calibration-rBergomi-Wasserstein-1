import numpy as np 
from utils import Cov_exact
from joblib import Parallel, delayed

class rBergomi_direct:
    def __init__(self, M, T, params, P, cores, loop, rand_seed):
        #Time discretization
        self.M = M #number of time steps 
        self.T = T
        self.tau = self.T/self.M
        self.grid = np.linspace(0, T, self.M+1)        
        self.P = P #number of paths to generate 
        
        #Rough Bergomi model parameters 
        self.X0 = params["X0"]
        self.V0 = params["V0"]
        self.xi = params["xi"]
        self.nu = params["nu"]
        self.rho = params["rho"]
        self.H = params["H"]
        self.r = params["r"]
        
        #Precomputation
        self.cov = Cov_exact(self.M, self.H, self.T, self.rho)
        self.minue = self.nu**2/2 * self.grid[1:] **(2*self.H) #(M,) 1-d array  

         # generate the stock price paths in parallel
        self.num_cores = cores
        self.my_loops = loop             
        self.seed = rand_seed 
    
    
    def W_Z(self, chunk_size): 
        # return samples of BM increment, Volterra process respectively       
        WZ = np.random.multivariate_normal(np.zeros(2*self.M), self.cov, chunk_size)
        W = WZ[:, :self.M]
        Z = WZ[:, self.M:]
        dZ = Z - np.c_[np.zeros(chunk_size), Z][:, :-1]
        return dZ, W 
    
    # only the final stock price is outputed
    def generate_paths_chunk_final(self, chunk_size):        
        dZ, W = self.W_Z(chunk_size)
        V_chunk = self.xi * np.exp(self.nu * W - self.minue)
        X_chunk = self.X0 + (self.r - 0.5 * self.V0)*self.tau + np.sqrt(self.V0)*dZ[:,0]
        for j in range(1, self.M):
            X_chunk = X_chunk + (self.r - 0.5 * V_chunk[:, j-1]) * self.tau + np.sqrt(V_chunk[:, j-1]) * dZ[:, j]
        
        return np.exp(X_chunk)
    

    # Generate the final stock price in parallel
    def S_final(self): 
        np.random.seed(self.seed)
        my_S = np.zeros((1))       
        chunk_size = int(np.ceil(self.P / self.num_cores/ self.my_loops))
        
        for i in range(self.my_loops):        
            S_chunks = Parallel(n_jobs=self.num_cores)(delayed(self.generate_paths_chunk_final)(chunk_size) for j in range(self.num_cores))

            # Concatenate the path chunks to form the final path array
            S_p = np.concatenate(S_chunks)            
            
            my_S = np.concatenate((my_S, S_p))    
        
        return my_S[1:] 
    
    # only the stock price at the given maturities is outputed 
    # to facilitate the computation of implied volatility surface 
    def generate_paths_chunk_maturities(self, chunk_size, num_maturity):
        X_chunk = np.zeros((chunk_size, num_maturity))
        dZ, W = self.W_Z(chunk_size)
        V_chunk = self.xi * np.exp(self.nu * W - self.minue)
        i = 1
        start = self.X0 + (self.r - 0.5 * self.V0)*self.tau + np.sqrt(self.V0)*dZ[:,0]
        for j in range(1, self.M):            
            start = start + (self.r - 0.5 * V_chunk[:, j-1]) * self.tau + np.sqrt(V_chunk[:, j-1]) * dZ[:, j]
            if j == i * int(self.M / num_maturity) - 1:
                X_chunk[:, i-1] = start
                i += 1 
        return np.exp(X_chunk)
    
    def S_maturites(self, num_maturity):
        np.random.seed(self.seed)
        my_S = np.zeros(num_maturity).reshape(1, -1)
        chunk_size = int(np.ceil(self.P / self.num_cores/ self.my_loops))
        
        for i in range(self.my_loops):        
            S_chunks = Parallel(n_jobs=self.num_cores)(delayed(self.generate_paths_chunk_maturities)(chunk_size, num_maturity) for j in range(self.num_cores))

            # Concatenate the path chunks to form the final path array
            S_p = np.concatenate(S_chunks)            
            
            my_S = np.concatenate((my_S, S_p), axis = 0)    
        
        return my_S[1:, :]  
    
    # output the whole sample paths 
    def generate_paths_chunk(self, chunk_size): 
        X_chunk = np.zeros((chunk_size, self.M))      
        dZ, W = self.W_Z(chunk_size)
        V_chunk = self.xi * np.exp(self.nu * W - self.minue)
        X_chunk[:, 0] = self.X0 + (self.r - 0.5 * self.V0)*self.tau + np.sqrt(self.V0)*dZ[:,0]
        for j in range(1, self.M):
            X_chunk[:, j] = X_chunk[:, j-1] + (self.r - 0.5 * V_chunk[:, j-1]) * self.tau + np.sqrt(V_chunk[:, j-1]) * dZ[:, j]
        
        return np.exp(X_chunk)
    
    
    # Generate the paths of stock price in parallel
    def S_(self):
        np.random.seed(self.seed)
        my_S = np.zeros(self.M).reshape(1, -1)
        
        chunk_size = int(np.ceil(self.P / self.num_cores/ self.my_loops))
        
        for i in range(self.my_loops):        
            S_chunks = Parallel(n_jobs=self.num_cores)(delayed(self.generate_paths_chunk)(chunk_size) for j in range(self.num_cores))

            # Concatenate the path chunks to form the final path array
            S_p = np.concatenate(S_chunks)            
            
            my_S = np.concatenate((my_S, S_p), axis = 0)    
        
        return my_S[1:, :]  

    
    def European_price(self, K):
        S_final = self.S_final()
        payoff = S_final.reshape((-1, 1)) - K.reshape((1, -1))
        payoff[payoff < 0] = 0
        price = np.mean(np.exp(-self.r * self.T) * payoff, axis = 0)        
        return price
    



