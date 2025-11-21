import time 
import tensorflow as tf
from solver import * 

class Config(object):    
    X0 = 0.0
    V0 = 0.09
    r = 0
    
    M = 1000
    P = 2**14
    """
    M = 100
    P = 2**14
    """    
    seed = 8
    reps = 1e-4 
    T = 1
    N = 10 
    xi = 0.15
    H = 0.12
    rho = -0.7
    nu = 1.5
    
    # hyperparameters   
    num_iterations = 2000
    tolerance = 1e-4
    patience = 80
    min_delta = 1e-5
    



def train(config, true_price_surface, true_S):    
    model = Solver(config, true_price_surface, true_S)
    optimizer = optimizer_wass()
    
    loss_history = []
    param_history = []
    param_history.append([model.m_param.xi.numpy(), model.m_param.H.numpy(), model.m_param.rho.numpy(), model.m_param.nu.numpy()])
    
    wait = 0
    best_loss = float('inf')
    start = time.time()

    for step in range(1, config.num_iterations+1):       
        
        # Backpropagation
        with tf.GradientTape() as tape:            
            loss = model.loss_wasserstein1()

        # update all the trainable parameters
        params = model.trainable_variables       
        grad = tape.gradient(loss, params)
        del tape
        optimizer.apply_gradients(zip(grad, params))

        elapsed_time = time.time() - start
        loss_history.append([loss, elapsed_time])
        xi = model.m_param.xi
        H = model.m_param.H 
        rho = model.m_param.rho
        nu = model.m_param.nu
        param_history.append([xi.numpy(), H.numpy(), rho.numpy(), nu.numpy()])
        
        np.save("FinalResult/wass_loss_history_0.npy", np.array(loss_history))
        np.save("FinalResult/wass_param_history_0.npy", np.array(param_history))
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
    index = 0
    true_price_surface = np.load(f"truePrices/true_price_surface_{index}.npy") 
    true_S = np.load(f"truePrices/true_S_{index}.npy")
    train(config, true_price_surface, true_S)
    

        

        