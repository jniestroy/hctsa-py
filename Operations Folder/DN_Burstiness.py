def DN_Burstiness(y):
    r = np.std(y) / y.mean()
    B = ( r - 1 ) / ( r + 1 )
    return(B)
