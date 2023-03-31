#import numpy as np

import numpy as nnp
import jax.numpy as np
from jax import grad, jit, vmap, pmap

@jit
def convection_conserved(psiCurrent_hat, w1_hat, Kx, Ky):
    # Convservative form
    u1_hat = -(1j*Ky)*psiCurrent_hat
    v1_hat = (1j*Kx)*psiCurrent_hat
    w1 = np.real(np.fft.ifft2(w1_hat))
    conu1 = 1j*Kx*np.fft.fft2((np.real(np.fft.ifft2(u1_hat))*w1))
    conv1 = 1j*Ky*np.fft.fft2((np.real(np.fft.ifft2(v1_hat))*w1))
    convec_hat = conu1 + conv1

    # Non-conservative form
    w1x_hat = 1j*Kx*w1_hat
    w1y_hat = 1j*Ky*w1_hat
    conu1 = np.fft.fft2(np.real(np.fft.ifft2(u1_hat))*np.real(np.fft.ifft2(w1x_hat)))
    conv1 = np.fft.fft2(np.real(np.fft.ifft2(v1_hat))*np.real(np.fft.ifft2(w1y_hat)))
    convecN_hat = conu1 + conv1

    convec_hat = 0.5*(convec_hat + convecN_hat)

    return convec_hat

