def DN_CustomSkewness(y,whatSkew = 'pearson'):
    if whatSkew == 'pearson':
        return (3*np.mean(y) - np.median(y)) / np.std(y)
    elif whatSkew == 'bowley':
        qs = np.quantile(y,[.25,.5,.75])
        return (qs[2] + qs[0] - 2*qs[1]) / (qs[2] - qs[0])
    else:
         raise Exception('whatSkew must be either pearson or bowley. whatSkew: ' + str(whatSkew))
