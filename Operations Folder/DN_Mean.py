import numpy as np
import numba

@numba.jit(nopython=True,parallel=True)
def DN_Mean(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(y.mean())
