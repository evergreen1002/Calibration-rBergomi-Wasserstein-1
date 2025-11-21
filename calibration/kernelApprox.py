import numpy as np
import tensorflow as tf
import tensorflow.math as tfm 
import math
from poly_roots import roots_tf


class quadrature_tf:
    """
    tensorflow version to compute the parameters of mSOE scheme
    output：Lambda: 1-d tensor of nodes, dtype = tf.float64
            Omega: 1-d tensor of weights, dtype = tf.float64
            Nexp: scalar tensor of number of summation terms, dtype = tf.int32
    """
    def __init__(self, beta, reps, dt, Tfinal):
        """
        :param beta : the power of the power function 1/t^beta
        :param reps : desired relative error
        :[dt, Tfinal] : the interval on which the power function is approximated
        """
        self.beta = beta
        self.dtype = tf.float64
        self.reps = tf.cast(reps, dtype = self.dtype)
        self.dt = tf.cast(dt, dtype = self.dtype)
        self.Tfinal = tf.cast(Tfinal, dtype = self.dtype)
    
        delta = tf.cast(dt/Tfinal, dtype = self.dtype)
        self.h = 2*tf.cast(math.pi, dtype = self.dtype)/(tfm.log(tf.cast(3, dtype =self.dtype))+self.beta*tfm.log(1/tfm.cos(tf.cast(1, dtype = self.dtype)))+tfm.log(1/self.reps))
        # tlower = 1/self.beta*np.log(self.reps*gamma(1+self.beta))
        tlower = 1/self.beta*tfm.log(self.reps*tfm.exp(tfm.lgamma(1+self.beta)))
    
        # if beta >= 1:
        # tupper = np.log(1/delta)+np.log(np.log(1/self.reps))+tf.math.log(beta)+1/2
        # else:
        tupper = tfm.log(1/delta)+tfm.log(tfm.log(1/self.reps))
    
        self.M = tf.math.floor(tlower/self.h)
        self.N = tf.cast(tf.math.ceil(tupper/self.h), dtype = tf.int32)
    
        n1 = tf.range(start = self.M, limit =0, dtype = self.dtype)
        self.xs1 = -tf.math.exp(self.h * n1)
        self.ws1 = self.h/tf.math.exp(tf.math.lgamma(self.beta)) * tf.math.exp(self.beta*self.h*n1)
    
    
    def myls(self, A, b, eps=1e-12):
        """
        solve the rank deficient least squares problem by SVD
        return: x: the LS solution
        return: res: the residue
        """

        n = tf.shape(A)[1]
        S, U, V = tf.linalg.svd(A, full_matrices=False)
        b = tf.cast(b, dtype = U.dtype)
        
        r = tf.reduce_sum(tf.cast(S>eps, dtype = tf.int64))
        # size (n, 1)
        x = tf.zeros([n,1], dtype = tf.complex128)
        for i in range(r):
            x = x + tf.reduce_sum((U[:, i] * b))/tf.cast(S[i], dtype=U.dtype) * tf.reshape(V[:, i], [-1, 1])
        
        res = tf.norm(tf.matmul(A, x) - tf.reshape(b, [-1, 1]))/tf.norm(b)
        return x, res
            
  
 
    
    def myls2(self, A, b, eps=1e-13):
        """
        solve the rank deficient least squares problem by SVD
        return: x: the LS solution
        return: res: the residue
        """
        m = tf.squeeze(tf.cast(tf.math.abs(self.M), dtype = tf.int64))
        # n = A.shape[1]
        Q, R = tf.linalg.qr(A)
        s = tf.linalg.diag_part(R)
        r = tf.squeeze(tf.reduce_sum(tf.cast(tf.math.abs(s)>eps, tf.int64)))
        Q = Q[:, :r]
        R = R[:r, :r]
        b1 = b[r:m+r]
        # size (r, 1)
        x = tf.linalg.solve(R, tf.linalg.matmul(tf.transpose(Q), tf.reshape(b1, [-1, 1])))
        # size (r, )
        x = tf.squeeze(x)
        
        return x
    
    def prony(self, xs, ws):
        """
        Reduce the number of quadrature points by Prony's method
        """
        M = tf.squeeze(tf.cast(tf.math.abs(self.M), dtype = tf.int64))
        errbnd = 1e-12

        h = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)
        for i in range(2*M):
            h = h.write(tf.cast(i, tf.int32), tf.reduce_sum(xs**(tf.cast(i, dtype = self.dtype)) * ws))
        h = h.stack()


        H = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)
        for i in range(M):
            H = H.write(tf.cast(i, tf.int32), h[i : i+M])
        # size (M, M)
        H = H.stack()

        b = -h
        q = self.myls2(H, b, errbnd)

        Coef = tf.concat([tf.constant(np.array([1]), dtype = self.dtype), tf.reverse(q, axis = [0])], axis = 0)
        xsnew = roots_tf(Coef)
        B = tf.TensorArray(xsnew.dtype, size=0, dynamic_size=True, clear_after_read=False)
        for i in range(2*M):
            B = B.write(tf.cast(i, tf.int32), xsnew ** i)
        B = B.stack()
        
        wsnew, res = self.myls(B, h, errbnd)
        ind = tf.where(tf.math.real(xsnew) >=0)
        p = tf.shape(ind)[0]
        assert tf.reduce_sum(tf.cast(tf.math.abs(tf.gather(wsnew, ind)) < 1e-15,dtype = tf.int32)) == p
        
        ind = tf.where(tf.math.real(xsnew)<0)
        xsnew = tf.squeeze(tf.gather(xsnew, ind))
        wsnew = tf.squeeze(tf.gather(wsnew, ind))
        
        return wsnew, xsnew


    def main(self):
        ws1new, xs1new = self.prony(self.xs1, self.ws1)
        n2 = tf.cast(tf.linspace(0, self.N, self.N+1), dtype = self.dtype)
        xs2 = -tf.math.exp(self.h * n2)
        ws2 = self.h/tf.math.exp(tf.math.lgamma(self.beta)) * tf.math.exp(self.beta * self.h * n2)
        xs = tf.concat([-tf.math.real(xs1new), tf.squeeze(-tf.math.real(xs2))], axis = 0)
        ws = tf.concat([tf.math.real(ws1new), tf.squeeze(tf.math.real(ws2))], axis = 0)
        
        xs = xs/self.Tfinal
        ws = ws/self.Tfinal **self.beta
        nexp = tf.shape(ws)[0]
        
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
        rerr = tf.norm((ftrue - fcomp) /ftrue, np.inf)
        print('The actual relative L_inf error is', rerr)



