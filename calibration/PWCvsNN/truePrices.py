import numpy as np
import tensorflow as tf
from kernelApprox import *
from rBergomi_mSOE_tf import *


class Config(object):
    X0 = 0.0    
    r = 0
    M = 1500
    P = 2**17
    seed = 8
    reps = 1e-4
    T = 1.5
    N = 15
    datype = tf.float64
    

def trueP(m_param, xi, config):
    train_list = [3, 5, 10, 12, 15]
    T_list = tf.constant([0.3, 0.5, 1.0, 1.2, 1.5], dtype = config.datype)
    K = tf.constant([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15], dtype=config.datype)
    
    params = {"X0": config.X0,
              "V0": xi[:, 0],
              "xi": xi[:, 1:], 
              "H": m_param[1],
              "rho": m_param[2],
              "nu": m_param[3],
              "r": config.r}
    quad = quadrature_tf(0.5 - params["H"], config.reps, config.T/config.M, config.T)
    Lambda, Omega, _= quad.main()
    mSOE = rBergomi_mSOE(config.M, config.T, params, config.P, Lambda, Omega)
    tf.random.set_seed(config.seed)
    S_maturity = mSOE.S_maturities(config.N)
    true_S = []
    for l in train_list:
        true_S.append(S_maturity[:, l-1])
    
    true_S = tf.stack(true_S, axis = 1)
    
    true_price_surface = []
    for i in range(len(train_list)):
        S_final = true_S[:, i]
        payoff_put = tf.reshape(K[:4], [1, -1]) - tf.reshape(S_final, [-1, 1])
        payoff_put = tf.where(payoff_put < 0, x = 0, y = payoff_put)
        payoff_call = tf.reshape(S_final, [-1, 1]) - tf.reshape(K[4:], [1, -1])
        payoff_call = tf.where(payoff_call < 0, x = 0, y = payoff_call)
        price_put = tf.math.reduce_mean(tf.math.exp(-config.r * T_list[i]) * payoff_put, axis = 0)
        price_call = tf.math.reduce_mean(tf.math.exp(-config.r * T_list[i]) * payoff_call, axis = 0)
        price = tf.concat([price_put, price_call], axis = 0)
        true_price_surface.append(price)
    
    true_price_surface = tf.stack(true_price_surface, axis = 0)

    return true_S[:, :3], true_price_surface


config = Config()
m_param = tf.constant([0.09, 0.07, -0.9, 1.9], dtype = config.datype)
steps = tf.reshape(tf.linspace(0.0, config.T, config.M + 1), [1, -1])
xi_list = []

xi_list.append(0.05 * tf.math.exp(-steps))
xi_list.append(0.02 + 0.03 * tf.math.exp(-5*steps) + 3 * steps * tf.math.exp(-5*steps))
xi_list.append(0.03 + 0.05 * tf.math.exp(-5 * (steps - 0.3)**2) + 0.01 * tf.math.sin(15 * steps))
xi_list = tf.stack(xi_list, axis = 0)


for i in range(3):
    true_S, true_price_surface = trueP(m_param, xi_list[i, :, :], config)
    true_S = tf.sort(true_S, 0)
    np.save(f"truePrices/true_S_{i+1}.npy", true_S.numpy())
    np.save(f"truePrices/true_price_surface_{i+1}.npy", true_price_surface.numpy())



    
    


    