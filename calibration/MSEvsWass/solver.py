import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers import Adam
from rBergomi_mSOE_tf import *
from kernelApprox import * 
 

    
# class of trainable objects: wrapped by keras layer
# regard all the model parameters as trainable scalar variables
class model_param(tf.keras.layers.Layer):
    def __init__(self, config):
        super(model_param, self).__init__()
        self.init_xi = tf.keras.initializers.Constant(value = config.xi)
        self.init_H = tf.keras.initializers.Constant(value = config.H)
        self.init_rho = tf.keras.initializers.Constant(value = config.rho)
        self.init_nu = tf.keras.initializers.Constant(value = config.nu)
        self.xi = self.add_weight(initializer= self.init_xi, trainable= True, name='m_xi')
        self.H = self.add_weight(initializer= self.init_H, trainable= True, name='m_H')
        self.rho = self.add_weight(initializer=self.init_rho, trainable= True, name='m_rho')
        self.nu = self.add_weight(initializer=self.init_nu, trainable= True, name='m_nu')

    def call(self):
        pass


class Solver(tf.keras.Model):
    def __init__(self, config, true_price_surface, true_S):
        super(Solver, self).__init__()        
       
        self.config = config
        self.dt = self.config.T / self.config.M        
        self.m_param = model_param(config)
        self.datype = tf.float64        
        self.train_list = [3, 5, 10]        
        self.true_price_surface = tf.constant(true_price_surface, dtype = self.datype)
        self.true_S = tf.constant(true_S, dtype = self.datype)      

    
    def loss_mse(self):        
        quad = quadrature_tf(0.5 - self.m_param.H, self.config.reps, self.dt, self.config.T)
        Lambda, Omega, _ = quad.main()        
        params = {"X0": self.config.X0, "V0": self.config.V0, "xi": self.m_param.xi,\
                   "H": self.m_param.H, "rho": self.m_param.rho, "nu": self.m_param.nu, "r": self.config.r}        
        mSOE = rBergomi_mSOE(self.config.M, self.config.T, params, self.config.P, Lambda, Omega)
        tf.random.set_seed(self.config.seed)
        S_maturity = mSOE.S_maturities(self.config.N)        
        price_surface = []
        K = tf.constant([0.9, 0.95, 1, 1.05], dtype = self.datype)
        
        
        for i in self.train_list:
            S_final = S_maturity[:, i-1]            
            payoff_put = tf.reshape(K[:2], [1,-1]) - tf.reshape(S_final, [-1, 1])
            payoff_put = tf.where(payoff_put < 0, x = 0, y = payoff_put)

            payoff_call = tf.reshape(S_final, [-1, 1]) - tf.reshape(K[2:], [1, -1])
            payoff_call = tf.where(payoff_call < 0, x = 0, y = payoff_call)
            price_put = tf.math.reduce_mean(tf.cast(tf.math.exp(-self.config.r * i/10), self.datype)* payoff_put, axis = 0)
            price_call = tf.math.reduce_mean(tf.cast(tf.math.exp(-self.config.r * i/10), self.datype) * payoff_call, axis = 0)
            
            price = tf.concat([price_put, price_call], axis = 0)
            price_surface.append(price)
            
        price_surface = tf.stack(price_surface, axis = 0) 
        true_train_price = self.true_price_surface[:, 2:6]
        
        return tf.math.reduce_mean((true_train_price - price_surface)**2)
    
    
    def loss_wasserstein1(self):
        quad = quadrature_tf(0.5 - self.m_param.H, self.config.reps, self.dt, self.config.T)
        Lambda, Omega, _ = quad.main()        
        params = {"X0": self.config.X0, "V0": self.config.V0, "xi": self.m_param.xi,\
                   "H": self.m_param.H, "rho": self.m_param.rho, "nu": self.m_param.nu, "r": self.config.r}        
        mSOE = rBergomi_mSOE(self.config.M, self.config.T, params, self.config.P, Lambda, Omega)
        tf.random.set_seed(self.config.seed)
        S_maturity = mSOE.S_maturities(self.config.N)          

        train_S = []    
        for l in self.train_list:
            train_S.append(S_maturity[:, l-1])    
        train_S = tf.stack(train_S, axis = 1)
        train_S = tf.sort(train_S, 0)
        w_1 = tf.math.reduce_mean(tf.math.abs(self.true_S - train_S), axis = 0)

        return tf.math.reduce_mean(tf.math.exp(-self.config.r * tf.convert_to_tensor(self.train_list, dtype = self.datype)/10) * w_1)
    
def optimizer_wass():
    boundaries = [800]
    values = [1e-3, 2e-4]
    lr_schedule = PiecewiseConstantDecay(boundaries, values)    
    optimizer = Adam(learning_rate = lr_schedule)
    return optimizer

def optimizer_mse():
    boundaries = [20]
    values = [3e-3, 1e-3]
    lr_schedule = PiecewiseConstantDecay(boundaries, values)
    optimizer = Adam(learning_rate = lr_schedule)
    return optimizer 