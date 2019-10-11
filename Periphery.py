import numpy as np
def BF_sgnchange(y,doFind = 0):
    if doFind == 0:
        return (np.multiply(y[1:],y[0:len(y)-1]) < 0)
    indexs = np.where((np.multiply(y[1:],y[0:len(y)-1]) < 0))
    return indexs

def BF_makeBuffer(y, bufferSize):

    N = len(y)

    numBuffers = int(np.floor(N / bufferSize))

    y_buffer = y[0:numBuffers*bufferSize]

    y_buffer = y_buffer.reshape((numBuffers,bufferSize))

    return y_buffer

def BF_embed(y,tau = 1,m = 2,makeSignal = 0,randomSeed = [],beVocal = 0):

    N = len(y)

    N_embed = N - (m - 1)*tau

    if N_embed <= 0:
        raise Exception('Time Series (N = %u) too short to embed with these embedding parameters')
    y_embed = np.zeros((N_embed,m))

    for i in range(1,m+1):

        y_embed[:,i-1] = y[(i-1)*tau:N_embed+(i-1)*tau]
    return(y_embed)

def BF_iszscored(x):
    numericThreshold = 2.2204E-16
    iszscored = ((np.absolute(np.mean(x)) < numericThreshold) & (np.absolute(np.std(x)-1) < numericThreshold))
    return(iszscored)

