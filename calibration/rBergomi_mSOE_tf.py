import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def Cov_SOE_tf(N, Lambda, tau, H):
    """
    Covariance matrix for mSOE scheme
    :param N: number of summation terms
    :param Lambda: an array contains the nodes with size (N,)
    :param tau: time step size
    :param H: Hurst index
    return: the covariance matrix with size (N+2, N+2)
    """
    cov = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)
    row_1 = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)

    # for 1st row
    row_1 = row_1.write(0, tau)
    for i in range(N):
        a = 1/Lambda[i] * (1 - tf.math.exp(-Lambda[i] * tau))
        row_1 = row_1.write(i+1, a)
    row_1 = row_1.write(N+1, tf.math.sqrt(2 * H) / (H + 0.5) * tau **(H + 0.5))
    row_1 = row_1.stack()
    cov = cov.write(0, row_1)

    # for 2nd to N+1 th row
    for j in range(N):
        row_j = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)
        a = 1/Lambda[j] * (1 - tf.math.exp(-Lambda[j] * tau))
        row_j = row_j.write(0, a)
        for i in range(N):
            lam_sum = Lambda[j] + Lambda[i]
            b = 1/lam_sum * (1 - tf.math.exp(-lam_sum * tau))
            row_j = row_j.write(i+1, b)
        c = tf.math.sqrt(2 * H)/(Lambda[j] ** (H + 0.5)) * tf.math.igamma(H + 0.5, Lambda[j] * tau)\
        * tf.math.exp(tf.math.lgamma(H + 0.5))
        row_j = row_j.write(N+1, c)
        row_j = row_j.stack()
        cov = cov.write(j+1, row_j)

    # for last row
    row_f = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)
    row_f = row_f.write(0, tf.math.sqrt(2 * H) / (H + 0.5) * tau **(H + 0.5))
    for i in range(N):
        a = tf.math.sqrt(2 * H)/Lambda[i] ** (H + 0.5) * tf.math.igamma(H + 0.5, Lambda[i] * tau)\
        * tf.math.exp(tf.math.lgamma(H + 0.5))
        row_f = row_f.write(i+1, a)
    row_f = row_f.write(N+1, tau**(2*H))
    row_f = row_f.stack()
    cov = cov.write(N+1, row_f)
    cov = cov.stack()
    return cov


class rBergomi_mSOE:
    """
    tensorflow version to solve the rBergomi model by mSOE scheme
    """
    def __init__(self, M, T, params, P, Lambda, Omega):
        #Time discretization
        self.M = M # number of time intervals 
        self.T = T # expiration           
        self.P = P #number of paths to generate 
        self.Lambda = Lambda # 1-d tensor
        self.Omega = Omega # 1-d tensor
        self.Nexp = tf.shape(self.Lambda)[0] # scalar tensor
        self.dtype = tf.float64
        self.tau = tf.cast(self.T / self.M, dtype = self.dtype)
        self.grid = tf.cast(tf.linspace(0, T, self.M + 1), dtype = self.dtype)
        
        #Rough Bergomi model parameters 
        self.X0 = tf.cast(params["X0"], dtype = self.dtype)
        self.V0 = tf.cast(params["V0"], dtype = self.dtype)
        self.r = tf.cast(params["r"], dtype = self.dtype)
        self.xi = params["xi"] # scalar tensor
        self.nu = params["nu"] # scalar tensor
        self.rho = params["rho"] # scalar tensor
        self.H = params["H"] # scalar tensor
        
              
        #Precomputation 
        # size = (1, Nexp)
        self.coef = tf.math.exp(-self.tau * tf.reshape(self.Lambda, [1, -1]))
        self.minue = tf.reshape(self.nu**2/2 * (self.grid[1:])**(2* self.H), [1, -1])
        
        # compute covariance matrix 
        cov = Cov_SOE_tf(self.Nexp, self.Lambda, self.tau, self.H)

        # decomposition, "root" of covariance matrix
        S,U,V = tf.linalg.svd(cov)
        S_half = tf.math.sqrt(S + 1e-15)
        self.scale = tf.linalg.matmul(tf.linalg.matmul(U, tf.linalg.diag(S_half)), tf.transpose(V))
        self.normal_dist = tfp.distributions.Normal(loc = tf.zeros(1, dtype = self.dtype), scale = tf.ones(1, dtype = self.dtype))

        
    def multi_dist(self):
        normal_sample = tf.squeeze(self.normal_dist.sample(sample_shape = [self.P, self.Nexp + 2]))

        # size (self.P, Nexp + 2)
        return tf.linalg.matmul(normal_sample, self.scale)
    

    # generate volatility paths without the forward variance curve
    # generate the paths of Brownian motion that drives the stock price 
    def generate_V(self):        
        W = []
        mul = []
        hist = tf.zeros((self.P, self.Nexp), dtype = self.dtype)
        sample = self.multi_dist()
        W.append(sample[:, 0])
        mul.append(sample[:, -1])
        
        for i in range(2, self.M + 1):    
           
            # size = (self.P, Nexp)
            hist = (hist + sample[:, 1:-1]) * self.coef
            # size = (self.P, )
            hist_part = tf.math.sqrt(2*self.H) * tf.reduce_sum(tf.reshape(self.Omega, [1, -1]) * hist, axis = 1)
            sample = self.multi_dist()
            W.append(sample[:, 0])
            mul.append(sample[:, -1] + hist_part)
       
        W = tf.stack(W, axis = 1)
        mul = tf.stack(mul, axis = 1)
        V = tf.math.exp(self.nu * mul - self.minue) # (self.P, M)
        
        return V, W

    # only the stock price at the given maturities is outputed 
    # to facilitate the computation of implied volatility surface 
    def S_maturities(self, num_maturity):
        X = []
        V, W = self.generate_V()
        V = self.xi * V
        Z = self.rho * W + tf.math.sqrt((1 - self.rho**2) * self.tau) * tf.squeeze(self.normal_dist.sample(sample_shape = [self.P, self.M]))
        i = 1
        start = (self.X0 + (self.r - 0.5 * self.V0) * self.tau) * tf.ones(self.P, dtype = self.dtype) + tf.math.sqrt(self.V0) * Z[:, 0]
        for j in range(1, self.M):
            start = start + (self.r * tf.ones(self.P, dtype = self.dtype) - 0.5 * V[:, j-1]) * self.tau + tf.math.sqrt(V[:, j-1]) * Z[:, j]
            if j == i * int(self.M / num_maturity) -1:
                X.append(start)
                i += 1 
        X = tf.stack(X, axis = 1)

        return X
        

    

