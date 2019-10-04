def DN_Moments(y,theMom = 1):
    return stats.moment(y,theMom) / np.std(y)
