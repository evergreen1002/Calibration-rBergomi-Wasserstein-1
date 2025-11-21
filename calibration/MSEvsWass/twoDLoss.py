import numpy as np
import tensorflow as tf
from kernelApprox import *
from rBergomi_mSOE_tf import *

def landscape(m_param, config, true_price_surface, true_S):     
        train_list = [3, 5, 10]          
        T_list = tf.constant([0.3, 0.5, 1.0], dtype = config.datype)
        K = tf.constant([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15], dtype=config.datype) 
        quad = quadrature_tf(0.5 - m_param[1], config.reps, config.T/config.M, config.T)
        Lambda, Omega, _ = quad.main()        
        params = {"X0": config.X0, 
                  "V0": config.V0, 
                  "xi": m_param[0],
                  "H": m_param[1], 
                  "rho": m_param[2], 
                  "nu": m_param[3], 
                  "r": config.r}        
        mSOE = rBergomi_mSOE(config.M, config.T, params, config.P, Lambda, Omega)
        tf.random.set_seed(config.seed)
        S_maturity = mSOE.S_maturities(config.N)        
        price_surface = []
        
        for i in range(len(train_list)):
            S_final = S_maturity[:, train_list[i]-1]
            payoff_put = tf.reshape(K[:4], [1, -1]) - tf.reshape(S_final, [-1, 1])
            payoff_put = tf.where(payoff_put < 0, x = 0, y = payoff_put)
            payoff_call = tf.reshape(S_final, [-1, 1]) - tf.reshape(K[4:], [1, -1])
            payoff_call = tf.where(payoff_call < 0, x = 0, y = payoff_call)
            price_put = tf.math.reduce_mean(tf.math.exp(-config.r * T_list[i]) * payoff_put, axis = 0)
            price_call = tf.math.reduce_mean(tf.math.exp(-config.r * T_list[i]) * payoff_call, axis = 0)
            price = tf.concat([price_put, price_call], axis = 0)
            price_surface.append(price)
            
        price_surface = tf.stack(price_surface, axis = 0)  

        train_surface = price_surface[:, 2:6]  
        true_train_surface = true_price_surface[:, 2:6]
        mse = tf.math.reduce_mean((true_train_surface - train_surface)**2)
        
        train_S = []    
        for l in train_list:
            train_S.append(S_maturity[:, l-1])    
        train_S = tf.stack(train_S, axis = 1)
        train_S = tf.sort(train_S, 0)
        w_1 = tf.math.reduce_mean(tf.math.abs(true_S - train_S), axis = 0)
        wass = tf.math.reduce_mean(w_1)

        return wass, mse



class Config(object):
    X0 = 0.0
    V0 = 0.09
    r = 0
    M = 1000
    P = 2**14
    seed = 8
    reps = 1e-4
    T = 1
    N = 10
    datype = tf.float64

config = Config()

true_param = np.array([0.09, 0.07, -0.9, 1.9])
true_train_surface = np.load("truePrices/true_price_surface_0.npy")
true_S = np.load("truePrices/true_S_0.npy")

# 2D loss landscape
points = 25
wass_collection = np.zeros((4, 4, points, points))
mse_collection = np.zeros((4, 4, points, points))
range_collection = [
     np.linspace(0.03, 0.15, points),
     np.linspace(0.01, 0.13, points),
     np.linspace(-0.98, -0.5, points),
     np.linspace(1.3, 2.5, points)
]

for j in range(4):
    for i in range(j+1, 4):
        for k in range(points):
            for l in range(points):
                m_param = []
                for q in range(4):
                    if q == j:
                        m_param.append(range_collection[j][k])
                    elif q == i:
                        m_param.append(range_collection[i][l])
                    else:
                        m_param.append(true_param[q])
                
                wass_collection[i, j, k, l], mse_collection[i, j, k, l] = landscape(m_param, config, true_train_surface, true_S)

np.save("FinalResult/wass_collection_2D.npy", wass_collection)
np.save("FinalResult/mse_collection_2D.npy", mse_collection)
                



                    
                             
              