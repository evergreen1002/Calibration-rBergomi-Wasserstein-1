"""
Given the number of summation terms,
return the corresponding nodes and weights 
"""
from shidong import SHIDONGJ
def dichotomy_quad(beta, Nexp, dt, Tfinal):
    left = 1e-17
    right =  0.999999
    quad_1 = SHIDONGJ(beta, left, dt, Tfinal)
    _, _, n_1 = quad_1.main()
    quad_2 = SHIDONGJ(beta, right, dt, Tfinal)
    _, _, n_2 = quad_2.main()
    assert n_1 > Nexp and n_2 < Nexp
    
    while left < right:
        middle = (left + right)/2 
        quad = SHIDONGJ(beta, middle, dt, Tfinal)
        nodes, weights, n = quad.main()
        if n > Nexp:
            left = middle 
        elif n < Nexp:
            right = middle 
        else:
            return nodes, weights


N_list = [4, 8, 16]
dt = [1/128]
for i in N_list:    
    nodes, weights = dichotomy_quad(0.43, i, dt[0], 1)
    print(nodes)
    print(weights)

       


 
