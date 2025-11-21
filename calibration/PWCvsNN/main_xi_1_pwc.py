import time 
import tensorflow as tf
from solver import * 

class Config(object):    
    X0 = 0.0
    V0 = 0.05
    r = 0
     
    M = 1000
    P = 2**17
    """
    M = 80
    P = 2**14
    """
    
       
    seed = 8
    reps = 1e-4 
    T = 1
    N = 10  
    xi = 0.05 
    H = 0.07
    rho = -0.9
    nu = 1.9
    trainable = False

    num_pieces = 8

    beta_0 = 0.03
    beta_1 = 0.02
    beta_2 = 0.15
    ttau = 0.15

    num_hiddens = [4,4]
    alpha = 0.01
    
    # hyperparameters   
    num_iterations = 5000
    tolerance = 1e-4
    patience = 30
    min_delta = 1e-5
    datype = tf.float64


def train(config, true_S, type):    
    model = Solver(config, true_S, type)
    optimizer = optimizer_wass()
    
    loss_history = []
    param_history = []
    xi_history = []
    param_history.append([model.m_param.H.numpy(), model.m_param.rho.numpy(), model.m_param.nu.numpy()])
    
    wait = 0
    best_loss = float('inf')
    start = time.time()
    
    for step in range(1, config.num_iterations+1):       
        
        # Backpropagation
        with tf.GradientTape() as tape:            
            loss, xi = model.loss_wasserstein1()      
        

        # update all the trainable parameters
        params = model.trainable_variables       
        grad = tape.gradient(loss, params)
        del tape
        optimizer.apply_gradients(zip(grad, params))

        elapsed_time = time.time() - start
        loss_history.append([loss, elapsed_time])
        xi_history.append(xi.numpy())
        H = model.m_param.H 
        rho = model.m_param.rho
        nu = model.m_param.nu
        param_history.append([H.numpy(), rho.numpy(), nu.numpy()])        
        
        np.save("FinalResult/loss_history_1_pwc.npy", np.array(loss_history))
        np.save("FinalResult/param_history_1_pwc.npy", np.array(param_history))
        np.save("FinalResult/xi_history_1_pwc.npy", np.array(xi_history))
        
        print("After {i}th iteration".format(i = step), 
              "loss = {}".format(loss), 
              "xi = {xi}, H = {H}, rho = {rho}, nu = {nu}".format(xi = xi.numpy(),H = H.numpy(), rho = rho.numpy(), nu = nu.numpy()),\
               "---{} seconds elapse---".format(elapsed_time))        

        # The early stopping strategy:
        # condition 1: touches the tolerance
        if loss < config.tolerance:
            print(f'Iteration {step}: touches tolerance {loss:.2e} <= {config.tolerance}')
            break

        # condition 2: loss stably converges
        if loss < best_loss - config.min_delta:
            best_loss = loss
            wait = 0
        
        else:
            wait += 1
            if wait >= config.patience:
                print(f'Iteration {step}: loss stably converges at {loss:.2e}')
        


if __name__ == "__main__":
    config = Config()
    index = 1    
    true_S = np.load(f"truePrices/true_S_{index}.npy")
    train(config, true_S, type = "pwc")
    

        

        