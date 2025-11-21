import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import Constraint

from rBergomi_mSOE_tf import *
from kernelApprox import * 



# constrain the weight within the given bound 
class BoundConstraint(Constraint):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, w):
        return tf.clip_by_value(w, self.min_val, self.max_val)
    



# class of trainable objects: wrapped by keras layer
# regard [H, \rho, \nu] as trainable scalar variables
class model_param(tf.keras.layers.Layer):
    def __init__(self, config):
        super(model_param, self).__init__()          
        self.H = tf.constant(config.H)
        self.rho = tf.constant(config.rho)
        self.nu = tf.constant(config.nu)

    def call(self):
        raise NotImplementedError("Subclasses should implement this method")


# regard \xi_0(t) as a piecewise constant function
class model_param_pwc(model_param):
    def __init__(self, config):
        super(model_param_pwc, self).__init__(config)
        self.config = config
        self.init_xi = tf.keras.initializers.Constant(value = config.xi)
        self.xi = self.add_weight(shape = (1, config.num_pieces), initializer = self.init_xi, trainable = True, 
                                  name = "m_xi", dtype = config.datype, constraint=BoundConstraint(0.01, 0.3))
    
    def call(self, t):
        piece_length = int(self.config.M / self.config.num_pieces)
        xi_expanded = tf.repeat(self.xi, repeats=piece_length, axis=1)

        return xi_expanded




# take \xi_0(t) to follow the Nelson-Siegel model
class model_param_ns(model_param):
    def __init__(self, config):
        super(model_param_ns, self).__init__(config)
        self.init_beta_0 = tf.keras.initializers.Constant(value = config.beta_0)
        self.init_beta_1 = tf.keras.initializers.Constant(value = config.beta_1)
        self.init_beta_2 = tf.keras.initializers.Constant(value = config.beta_2)
        self.init_ttau = tf.keras.initializers.Constant(value = config.ttau)

        self.beta_0 = self.add_weight(initializer= self.init_beta_0, trainable= True, name='m_beta_0', 
                                      dtype = config.datype, constraint= BoundConstraint(1e-5, 0.5))
        self.beta_1 = self.add_weight(initializer=self.init_beta_1, trainable= True, name='m_beta_1', 
                                      dtype = config.datype, constraint= BoundConstraint(1e-5, 0.5))
        self.beta_2 = self.add_weight(initializer=self.init_beta_2, trainable= True, name='m_beta_2', 
                                      dtype = config.datype, constraint= BoundConstraint(1e-5, 1))
        self.ttau = self.add_weight(initializer=self.init_ttau, trainable= True, name='m_ttau', 
                                      dtype = config.datype, constraint= BoundConstraint(1e-3, 5))
    
    def call(self, t):
        return self.beta_0 + self.beta_1 * tf.math.exp(-t/self.ttau) + self.beta_2 * (t/self.ttau) * tf.math.exp(-t/self.ttau)



# take \xi_0(t) to follow the Nelson-Siegel + NN model
class model_param_nspnn(model_param_ns):
    def __init__(self, config):
        super(model_param_nspnn, self).__init__(config)
        self.init_alpha = tf.keras.initializers.Constant(value = config.alpha)
        self.alpha = self.add_weight(initializer= self.init_alpha, trainable= True, name='m_alpha',dtype = config.datype, 
                                     constraint = BoundConstraint(-0.5, 0.5))
        self.num_hiddens = config.num_hiddens
        self.dense_layers = [tf.keras.layers.Dense(self.num_hiddens[i],
                                                   use_bias=True,
                                                   kernel_initializer = 'glorot_uniform', 
                                                   bias_initializer = tf.initializers.Constant(value = 0.0), 
                                                   activation=tf.nn.leaky_relu,
                                                   dtype = config.datype)
                             for i in range(len(self.num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(1, 
                                                    kernel_initializer = 'glorot_uniform', 
                                                    bias_initializer = tf._initializers.Constant(value = config.xi),
                                                    activation = None, 
                                                    dtype = config.datype))
    
    def call(self, t):
        nn_ = tf.reshape(t, [-1, 1])
        for layer in self.dense_layers:
            nn_ = layer(nn_)
        nn_ = tf.reshape(nn_, [1,-1])

        ns_ = self.beta_0 + self.beta_1 * tf.math.exp(-t/self.ttau) + self.beta_2 * (t/self.ttau) * tf.math.exp(-t/self.ttau)
        return tf.math.abs(ns_ * (1 + self.alpha * tf.tanh(nn_)))
        
    

class Solver(tf.keras.Model):
    def __init__(self, config, true_S, type):
        """
        type: the parameterization type of the initial forward variance curve
        takes value in ["pwc", "ns", "ns+nn"]
        """
        super(Solver, self).__init__()        
       
        self.config = config        
        self.dt = self.config.T / self.config.M        
        self.type = type
        if type == "pwc":
            self.m_param = model_param_pwc(config)
        elif type == "ns":
            self.m_param = model_param_ns(config)
        elif type == "ns+nn":
            self.m_param = model_param_nspnn(config)
        
        self.datype = tf.float64        
        self.train_list = [3, 5, 10]
        self.true_S = tf.constant(true_S, dtype = self.datype) 
    
    
    def loss_wasserstein1(self):
        quad = quadrature_tf(0.5 - self.m_param.H, self.config.reps, self.dt, self.config.T)
        Lambda, Omega, _ = quad.main()   
        t_grid = tf.reshape(tf.linspace(0, self.config.T, self.config.M + 1)[1:], [1, -1])
        t_grid = tf.cast(t_grid, tf.float64) 
        xi = self.m_param.call(t_grid)  

        params = {"X0": self.config.X0, "V0": self.config.V0, "xi": xi,
                   "H": self.m_param.H, "rho": self.m_param.rho, 
                   "nu": self.m_param.nu, "r": self.config.r}        
        mSOE = rBergomi_mSOE(self.config.M, self.config.T, params, self.config.P, Lambda, Omega)
        tf.random.set_seed(self.config.seed)
        S_maturity = mSOE.S_maturities(self.config.N)          

        train_S = []    
        for l in self.train_list:
            train_S.append(S_maturity[:, l-1])    
        train_S = tf.stack(train_S, axis = 1)
        train_S = tf.sort(train_S, 0)
        w_1 = tf.math.reduce_mean(tf.math.abs(self.true_S - train_S), axis = 0)
        
        t_grid_extend = tf.reshape(tf.linspace(0.0, 1.5, 1501)[1:], [1, -1])
        t_grid_extend = tf.cast(t_grid_extend, tf.float64)
        xi_extend = self.m_param.call(t_grid_extend) 
        return tf.math.reduce_mean(tf.math.exp(-self.config.r * tf.convert_to_tensor(self.train_list, dtype = self.datype)/10) * w_1), xi_extend
    

def optimizer_wass():        
    boundaries = [200, 1000]
    values = [1e-3, 7e-4, 3e-4]
    lr_schedule = PiecewiseConstantDecay(boundaries, values)    
    optimizer = Adam(learning_rate = lr_schedule)
    return optimizer

