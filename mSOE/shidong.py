import numpy as np
from scipy.special import gamma
from scipy.linalg import hankel, qr, svd, norm

class SHIDONGJ:
    def __init__(self, beta, reps, dt, Tfinal):
        """
        :param beta : the power of the power function 1/t^beta
        :param reps : desired relative error
        :[dt, Tfinal] : the interval on which the power function is approximated
        """
        self.beta = beta 
        self.reps = reps 
        self.dt = dt    
        self.Tfinal = Tfinal
    
        delta = dt/Tfinal
        self.h = 2*np.pi/(np.log(3)+self.beta*np.log(1/np.cos(1))+np.log(1/self.reps)) 
        tlower = 1/self.beta*np.log(self.reps*gamma(1+self.beta))
    
        if beta >= 1:
            tupper = np.log(1/delta)+np.log(np.log(1/self.reps))+np.log(beta)+1/2
        else:
            tupper = np.log(1/delta)+np.log(np.log(1/self.reps))
    
        M = int(np.floor(tlower/self.h))
        self.N = int(np.ceil(tupper/self.h))
    
        n1 = np.arange(M, 0)
        self.xs1 = -np.exp(self.h * n1)
        self.ws1 = self.h/gamma(self.beta) * np.exp(self.beta*self.h*n1)
    
    
    def myls(self, A, b, eps=1e-12):
        """
        solve the rank deficient least squares problem by SVD
        return: x: the LS solution
        return: res: the residue
        """
        m = A.shape[0]
        n = A.shape[1]
        U, S, V = svd(A, full_matrices=False)
        
        r = np.sum(S>eps)
        x = np.zeros((n,1))
        for i in range(int(r)):
            x = x + np.sum((U[:, i] * b))/S[i] * np.transpose(V)[:, i].reshape((n,1))
        
        res = norm(np.matmul(A, x) - b.reshape(m,1))/norm(b)
        return x, res
            
  
 
    
    def myls2(self, A, b, eps=1e-13):
        """
        solve the rank deficient least squares problem by SVD
        return: x: the LS solution
        return: res: the residue
        """
        m = A.shape[0]
        n = A.shape[1]
        Q, R = qr(A, mode = 'economic')
        s = np.diag(R)
        r = np.sum(np.abs(s)>eps)
        Q = Q[:, :r]
        R = R[:r, :r]
        b1 = b[r:m+r]
        x = np.linalg.solve(R, np.matmul(np.transpose(Q), b1))
        
        return x
    
    def prony(self, xs, ws):
        """
        Reduce the number of quadrature points by Prony's method
        """
        M = xs.size 
        errbnd = 1e-12
        h = np.zeros(2*M)
        
        for i in range(2*M):
            h[i] = np.sum(xs**(i) * ws)
        
        C = h[:M]
        R = h[M-1: 2*M-1]
        H = hankel(C, R)
        
        b = -h
        q = self.myls2(H, b, errbnd)
        r = q.size 
        A = np.zeros((2*M, r))
        
        Coef = np.concatenate((np.array([1]), np.flipud(q)))
        xsnew = np.roots(Coef)
        
        for i in range(2*M):
            A[i,:] = xsnew ** i
        
        wsnew, res = self.myls(A, h, errbnd);
        
        ind = np.argwhere(np.real(xsnew) >=0)
        p = ind.size
        assert np.sum(np.abs(wsnew[ind]) < 1e-15) == p
        
        ind = np.argwhere(np.real(xsnew)<0)
        xsnew = xsnew[ind].flatten()
        wsnew = wsnew[ind].flatten()
        
        return wsnew, xsnew 
    
    def main(self):
        ws1new, xs1new = self.prony(self.xs1, self.ws1)
        n2 = np.linspace(0, self.N, self.N+1)
        xs2 = -np.exp(self.h * n2)
        ws2 = self.h/gamma(self.beta) * np.exp(self.beta * self.h * n2)
        xs = np.concatenate((-np.real(xs1new), -np.real(xs2)))
        ws = np.concatenate((np.real(ws1new), np.real(ws2)))
        
        xs = xs/self.Tfinal
        ws = ws/self.Tfinal **self.beta
        nexp = ws.size   
        
        return xs, ws, nexp
    
    def test(self):
        xs, ws, nexp = self.main()
        m = 10000
        estart = np.log10(self.dt)
        eend = np.log10(self.Tfinal)
        texp = np.linspace(estart, eend, m)
        t = 10 ** texp
        
        ftrue = 1/(t **self.beta)
        fcomp = np.zeros(ftrue.size)

        for i in range(m):
            fcomp[i] = np.sum(ws * np.exp(-t[i] * xs))
            
        fcomp = np.real(fcomp)
        rerr = norm((ftrue - fcomp) /ftrue, np.inf)
        print('The actual relative L_inf error is', rerr)


"""
beta = 0.5 - 0.001
reps = 0.999999
dt = 1e-3 
T = 1 
quad = SHIDONGJ(beta, reps, dt, T)
_, _, N = quad.main()
print(N)
"""
