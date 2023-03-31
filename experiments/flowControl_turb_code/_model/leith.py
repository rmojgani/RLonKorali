#import numpy as np

import numpy as nnp
import jax.numpy as np
from jax import grad, jit, vmap, pmap
import jax

@jit
def leith_cs(action, veRL, Kx, Ky, w1_hat, NX):

    #print('action is:', action_leith)
    if action != None:
    #    if self.veRL !=0:
        CL3 = veRL#action_leith[0]
    else:
        CL3 = 0.17**3# (Lit)

    w1x_hat = -(1j*Kx)*w1_hat
    w1y_hat = (1j*Ky)*w1_hat
    w1x = np.real(np.fft.ifft2(w1x_hat))
    w1y = np.real(np.fft.ifft2(w1y_hat))
    abs_grad_omega = np.mean(np.sqrt( w1x**2+w1y**2  ))
    # 
    delta3 = (2*np.pi/NX)**3
    ve = CL3*delta3*abs_grad_omega
    return ve


