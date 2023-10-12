# ----------------------------------------------------------------------
# From Py2D: bd83aef
# ----------------------------------------------------------------------

# Eddy Viscosity SGS Models for 2D Turbulence solver


#import numpy as np
import jax.numpy as np
from jax import jit

from py2d.convert import strain_rate_2DFHIT_spectral

#strain_rate_2DFHIT_spectral = jit(strain_rate_2DFHIT_spectral)

@jit
def Tau_eddy_viscosity(eddy_viscosity, Psi_hat, Kx, Ky):
    '''
    Calculate the eddy viscosity term (Tau) in the momentum equation 
    '''
    S11_hat, S12_hat, S22_hat = strain_rate_2DFHIT_spectral(Psi_hat, Kx, Ky)
    S11 = np.real(np.fft.ifft2(S11_hat))
    S12 = np.real(np.fft.ifft2(S12_hat))
    S22 = np.real(np.fft.ifft2(S22_hat))

    Tau11 = -2*eddy_viscosity*S11
    Tau12 = -2*eddy_viscosity*(S12)
    Tau22 = -2*eddy_viscosity*S22

    return Tau11, Tau12, Tau22
