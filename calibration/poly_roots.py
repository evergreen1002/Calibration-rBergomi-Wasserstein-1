import tensorflow as tf
import numpy as np



def roots_tf(p):
    """
    Return the roots of a polynomial with coefficients given in p
    based on the source code of np.roots

    If the length of `p` is n+1 then the polynomial is described by::

    p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    p: 1-d tensor
    """
    # return the index of non_zero entries
    non_zero = tf.squeeze(tf.where(p != 0))

    # Return an empty array if polynomial is all zeros
    if len(non_zero) == 0:
        return tf.constant(np.array([]), dtype=tf.complex128)

    # number of trailing_zeros
    trailing_zeros = len(p) - 1 - int(non_zero[-1])

    # strip leading and trailing zeros
    p = p[int(non_zero[0]): int(non_zero[-1]) + 1]    
    N = len(p)
    if N > 1:
        # build companion matrix and find its eigenvalues        
        A = tf.linalg.diag(tf.ones((N - 2,), dtype=tf.float64), k=-1)
        # remove first row
        A = A[1:, :]
        r_1 = tf.reshape(-p[1:] / p[0], [1, -1])

        A = tf.concat([r_1, A], axis=0)
        roots = tf.linalg.eigvals(A)
        
    else:
        roots = tf.constant(np.array([]), dtype=tf.complex128)

    roots = tf.concat((roots, tf.zeros(trailing_zeros, roots.dtype)), axis=0)
    return roots


