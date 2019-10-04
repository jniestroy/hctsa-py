def CO_AutoCorr(y,lag = 1,method = 'TimeDomianStat'):
    if not isinstance(y,np.ndarray):
        y = np.asarray(y)
    if method == 'TimeDomianStat':
        if lag == []:
            acf = [1]
            for i in range(1,len(y)-1):
                acf.append(np.corrcoef(y[:-lag],y[lag:])[0,1])
            return acf
        return(np.corrcoef(y[:-lag],y[lag:])[0,1])
    else:
        N = len(y)
        nFFT = 2**(math.ceil(math.log2(N)) + 1)
        F = np.fft.fft(y - y.mean(),nFFT)
        F = np.multiply(F,np.conj(F))
        acf = np.fft.ifft(F)
        acf = acf / acf[0]
        acf = acf.real
        if lag == []:
            return acf
        return acf[lag]
