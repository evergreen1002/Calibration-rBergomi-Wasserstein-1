import numpy as np
from rBergomi_mSOE import * 
from dichotomy import * 
import time

X0 = 0
V0 = 0.235 ** 2
xi = 0.235 ** 2 
nu = 1.9
H = 0.07
rho = -0.9
r = 0
params = {"X0": X0, "V0": V0, "xi": xi, "nu": nu, "rho": rho, "H": H, "r": r}
M = [128, 256, 512, 1024, 2048]
P = 2**24

T = 1
seed = 0
Nexp = [4, 8, 16, 32, 64]
mytime = np.zeros((len(M), len(Nexp)))
for i in range(len(M)):
    for j in range(len(Nexp)):
        Lambda, Omega = dichotomy_quad(0.5 - H, Nexp[j], T/M[i], T)
        start_time = time.time()
        mSOE = rBergomi_mSOE(M[i], T, params, P, Lambda, Omega, 16, 256, seed)        
        np.save("FinalResult/ST_mSOE_{m}_{N}.npy".format(m = M[i], N = Nexp[j]), mSOE.S_final())
        end_time = time.time()
        mytime[i][j] = end_time - start_time

np.save("FinalResult/mSOE_time.npy", mytime)
    