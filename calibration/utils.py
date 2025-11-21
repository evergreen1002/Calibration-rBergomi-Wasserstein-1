import numpy as np
from scipy.stats import norm
import scipy.special as sc
from scipy.special import gamma
from scipy.optimize import brentq




def Cov_exact(M, H, T, rho):
    """
    Covariance matrix for exact method
    :param M: number of time steps
    :param T: expiration
    :param H: Hurst index
    :param rho: correlation between Brownian motions
    return: the covariance matrix with size (2*M, 2*M)
    """
    tau = T/M
    cov = np.zeros((2 * M, 2 * M))
    my_array = (np.arange(M+1) * tau)**(H + 0.5)
    coef_1 = 2 * H * gamma(H + 0.5)/gamma(H + 1.5) 
    coef_2 = rho * np.sqrt(2 * H)/(H + 0.5)
    for i in range(M):
        for j in range(i+1): # j<=i
            cov[i,j] = coef_1 * ((j+1)*tau)**(H + 0.5) * ((i+1)*tau)**(H - 0.5) * sc.hyp2f1(0.5-H, 1, H+1.5, (j+1)/(i+1))
            cov[j,i] = cov[i,j] 
            
            cov[i + M, j + M] = tau * (j+1)
            cov[j + M, i + M] = cov[i + M, j + M]
            
            cov[i, j + M] = coef_2 * (my_array[i+1] - my_array[i-j])
            cov[j + M, i] = cov[i, j + M]
            
            cov[i + M, j] = coef_2 * my_array[j+1]
            cov[j, i + M] = cov[i + M, j]
    
    return cov 

def Cov_hybrid(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov



def Cov_mSOE(N, Lambda, tau, H):
    """
    Covariance matrix for mSOE scheme
    :param N: number of summation terms
    :param Lambda: an array contains the nodes with size (N,)
    :param tau: time step size
    :param H: Hurst index
    return: the covariance matrix with size (N+2, N+2)
    """
    cov = np.zeros((N+2, N+2))
    cov[0,0] = tau
    cov[N+1, 0] = np.sqrt(2*H)/(H + 0.5) * (tau ** (H + 0.5))
    cov[0, N+1] = cov[N+1, 0]
    cov[N+1, N+1] = tau**(2*H)
        
    for i in range(N):
        cov[0, i+1] = 1/Lambda[i] * (1 - np.exp(-Lambda[i] * tau))
        cov[i+1, 0] = cov[0, i+1]
        
    for i in range(N):
        for j in range(i+1):
            cov[i+1, j+1] = 1/(Lambda[i] +  Lambda[j]) * (1 - np.exp(-(Lambda[i] +  Lambda[j]) * tau))
            cov[j+1, i+1] = cov[i+1, j+1]
        
    for i in range(N):
        cov[N+1, i+1] = np.sqrt(2*H) * Lambda[i]**(-H - 0.5) * sc.gammainc(H + 0.5, Lambda[i] * tau) * sc.gamma(H + 0.5)
        cov[i+1, N+1] = cov[N+1, i+1]
        
    return cov

def Cov_SOE(N, Lambda, tau):
    """
    Covariance matrix for SOE scheme
    :param N: number of summation terms
    :param Lambda: an array contains the nodes with size (N,)
    :param tau: time step size    
    return: the covariance matrix with size (N+1, N+1)    
    """
    cov = np.zeros((N+1, N+1))
    cov[0,0] = tau
    for i in range(N):
        cov[0, i+1] = 1/Lambda[i] * (1 - np.exp(-Lambda[i] * tau))
        cov[i+1, 0] = cov[0, i+1]
    
    for i in range(N):
        for j in range(i+1):
            cov[i+1, j+1] = 1/(Lambda[i] +  Lambda[j]) * (1 - np.exp(-(Lambda[i] +  Lambda[j]) * tau))
            cov[j+1, i+1] = cov[i+1, j+1]
    
    return cov 
    



def fbm(H, m, P):
    """
    Generate samples of fractional Brownian motion with Hurst index H
    :param H: Hurst index
    :param m: number of time steps
    :param P: number of samples
    return: an array contains P samples with size (P, m)
    """
    H_2 = 2 * H
    grid = np.linspace(0, 1, m + 1)[1:]
    mean = np.zeros(m)
    cov = np.zeros([m, m])

    # covariance matrix
    for i in range(m):
        for j in range(m):
            cov[i, j] = 0.5 * (grid[i] ** H_2 + grid[j] ** H_2 - np.abs(grid[i] - grid[j]) ** H_2)


    return np.random.multivariate_normal(mean, cov, P)


def bs(S, K, sigma, T, r):
    """
    S: current stock price
    K: strike price 
    sigma: volatility
    T: expiration
    r: interest rate
    return: the price of call option
    """
    V = sigma**2 * T
    sv = np.sqrt(V)
    # d1 = np.log(F/K) / sv + 0.5 * sv
    d1 = 1/sv * (np.log(S/K) + r*T + V/2)
    d2 = d1 - sv
    P = S * norm.cdf(d1) -  K * np.exp(-r * T) * norm.cdf(d2)
    return P

def bsinv(P, S, K, T, r):
    """
    P: call option price 
    S: current stock price
    K: strike price 
    T: expiration
    r: interest rate
    return: the implied volatility
    """

    def error(s):
        return bs(S, K, s, T, r) - P
    s = brentq(error, 1e-9, 1e+9)
    return s


def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)


