import itertools
import numba

@numba.jit(nopython=True,parallel=True)
def EN_PermEn(y,m = 2,tau = 1):

    x = BF_embed(y,tau,m)


    Nx = x.shape[0]

    permList = perms(m)
    numPerms = len(permList)

    countPerms = np.zeros(numPerms)


    for j in range(Nx):
        ix = np.argsort(x[j,:])

        for k in range(numPerms):
            if not (permList[k,:] - ix).all() :
                countPerms[k] = countPerms[k] + 1
                break

    p = countPerms / Nx
    p_0 = p[p > 0]
    permEn = -sum(np.multiply(p_0,np.log2(p_0)))



    mFact = math.factorial(m)
    normPermEn = permEn / np.log2(mFact)

    out = {'permEn':permEn,'normPermEn':normPermEn}

    return out

def perms(n):
    permut = itertools.permutations(np.arange(n))
    permut_array = np.empty((0,n))
    for p in permut:
        permut_array = np.append(permut_array,np.atleast_2d(p),axis=0)

    return(permut_array)

def DN_Moments(y,theMom = 1):
    if np.std(y) != 0:
        return stats.moment(y,theMom) / np.std(y)
    else:
        return 0

#@numba.jit(nopython=True,parallel=True)
def DN_Withinp(x,p = 1,meanOrMedian = 'mean'):
    N = len(x)

    if meanOrMedian == 'mean':
        mu = np.mean(x)
        sig = np.std(x)
    elif meanOrMedian == 'median':
        mu = np.median(x)
        sig = 1.35*stats.iqr(x)
    else:
        raise Exception('Unknown meanOrMedian should be mean or median')
    return np.sum((x >= mu-p*sig) & (x <= mu + p*sig)) / N

#@numba.jit(nopython=True)
#Quantile function seems to be slower with numba
def DN_Quantile(y,q = .5):
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.quantile(y,q))

def DN_RemovePoints(y,removeHow = 'absfar',p = .99):

    if removeHow == 'absclose':
        i =  np.argsort(-np.absolute(y),kind = 'mergesort')
    elif removeHow == 'absfar':
        i = np.argsort(np.absolute(y),kind = 'mergesort')
    elif removeHow == 'min':
        i =  np.argsort(-y,kind = 'mergesort')
    elif removeHow == 'max':
        i = np.argsort(y,kind = 'mergesort')

    N = len(y)

    out = {}

    rKeep = np.sort(i[0:int(np.round(N*(1-p)))])
    y_trim = y[rKeep]

    #print(rKeep)


    acf_y = SUB_acf(y,8)
    acf_y_trim = SUB_acf(y_trim,8)

    out['fzcacrat'] = CO_FirstZero(y_trim,'ac')/CO_FirstZero(y,'ac')

    out['ac1rat'] = acf_y_trim[1]/acf_y[1]

    out['ac1diff'] = np.absolute(acf_y_trim[1]-acf_y[1])

    out['ac2rat'] = acf_y_trim[2]/acf_y[2]

    out['ac2diff'] = np.absolute(acf_y_trim[2]-acf_y[2])

    out['ac3rat'] = acf_y_trim[3]/acf_y[3]

    out['ac3diff'] = np.absolute(acf_y_trim[3]-acf_y[3])

    out['sumabsacfdiff'] = sum(np.absolute(acf_y_trim-acf_y))

    out['mean'] = np.mean(y_trim)

    out['median'] = np.median(y_trim)

    out['std'] = np.std(y_trim)

    if stats.skew(y) != 0:
        out['skewnessrat'] = stats.skew(y_trim)/stats.skew(y)

    out['kurtosisrat'] = stats.kurtosis(y_trim)/stats.kurtosis(y)

    return out

def SUB_acf(x,n):
    acf = np.zeros(n)
    for i in range(n):
        acf[i] = CO_AutoCorr(x,i-1,'Fourier')
    return acf

def DN_OutlierInclude(y,thresholdHow='abs',inc=.01):
    if not BF_iszscored(y):
        muhat, sigmahat = stats.norm.fit(y)
        y = (y - muhat) / sigmahat
        warnings.warn('DN_OutlierInclude y should be z scored. So just converted y to z-scores')
    N = len(y)
    if thresholdHow == 'abs':
        thr = np.arange(0,np.max(np.absolute(y)),inc)
        tot = N
    if thresholdHow == 'p':
        thr = np.arange(0,np.max(y),inc)
        tot = sum( y >= 0)
    if thresholdHow == 'n':
        thr = np.arange(0,np.max(-y),inc)
        tot = sum( y <= 0)
    msDt = np.zeros((len(thr),6))
    for i in range(len(thr)):
        th = thr[i]

        if thresholdHow == 'abs':
            r = np.where(np.absolute(y) >= th)
        if thresholdHow == 'p':
            r = np.where(y >= th)
        if thresholdHow == 'n':
            r = np.where(y <= -th)

        Dt_exc = np.diff(r)

        msDt[i,0] = np.mean(Dt_exc)
        msDt[i,1] = np.std(Dt_exc) / np.sqrt(len(r))
        msDt[i,2] = len(Dt_exc) / tot * 100
        msDt[i,3] = np.median(r) / (N/2) - 1
        msDt[i,4] = np.mean(r) / (N/2) -1
        msDt[i,5] = np.std(r) / np.sqrt(len(r))

        return msDt

@numba.jit(nopython=True,parallel=True)
def DN_Burstiness(y):
    r = np.std(y) / y.mean()
    B = ( r - 1 ) / ( r + 1 )
    return(B)

#@numba.jit(nopython=True,parallel=True)
#oddly this function slows down with numba
def DN_pleft(y,th = .1):

    p  = np.quantile(np.absolute(y - np.mean(y)),1-th)


    return p / np.std(y)

def CO_FirstZero(y,corrFun = 'ac'):
    acf = CO_AutoCorr(y,[],'Fourier')
    N = len(y)
    for i in range(1,N-1):
        if acf[i] < 0:
            return i
    return N

def DN_Fit_mle(y,fitWhat = 'gaussian'):
    if fitWhat == 'gaussian':
        phat = stats.norm.fit(y)
        out = {'mean':phat[0],'std':phat[1]}
        return out
    else:
        print('Use gaussian geometric not implemented yet')

def CO_FirstMin(y, minWhat = 'ac'):
    if minWhat == 'mi':
        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
        acf = IN_AutoMutualInfo(y,x,'gaussian')
    else:
        acf = CO_AutoCorr(y,[],'Fourier')
    N = len(y)

    for i in range(1,N-1):
        if i == 2 and (acf[2] > acf[1]):
            return 1
        elif (i > 2) and (acf[i-2] > acf[i-1]) and (acf[i-1] < acf[i]):
            return i-1
    return N

def DN_IQR(y):
    return stats.iqr(y)

def DN_CompareKSFit(x,whatDist = 'norm'):
    xStep = np.std(x) / 100
    if whatDist == 'norm':
        a, b = stats.norm.fit(x)
        peak = stats.norm.pdf(a,a,b)
        thresh = peak / 100
        xf1 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.norm.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(x)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.norm.pdf(xf2,a,b)


    #since some outliers real far away can take long time
    #should probably do pre-proccessing before functions
    if whatDist == "uni":

        a,b = stats.uniform.fit(x)
        peak = stats.uniform.pdf(np.mean(x),a,b-a)
        thresh = peak / 100
        xf1 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.norm.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(x)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.norm.pdf(xf2,a,b)

    #might over write y since changing x
    if whatDist == 'beta':
        scaledx = (x - np.min(x) + .01*np.std(x)) / (np.max(x)-np.min(x)+.02*np.std(x))
        xStep = np.std(scaledx) /100
        a = stats.beta.fit(scaledx)
        b = a[2]
        a = a[1]
        thresh = 1E-5
        xf1 = np.mean(scaledx)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.beta.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(scaledx)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.beta.pdf(xf2,a,b)
        x = scaledx


    kde = stats.gaussian_kde(x)
    test_space = np.linspace(np.min(x),np.max(x),1000)
    kde_est = kde(test_space)
    if whatDist == 'norm':
        ffit = stats.norm.pdf(test_space,a,b)
    if whatDist == 'uni':
        ffit = stats.uniform.pdf(test_space,a,b-a)
    if whatDist == 'beta':
        ffit = stats.beta.pdf(test_space,a,b)

    out = {}

    out['adiff'] = sum(np.absolute(kde_est - ffit)*(test_space[1]-test_space[0]))

    out['peaksepy'] = np.max(ffit) - np.max(kde_est)

    r = (ffit != 0)

    out['relent'] = sum(np.multiply(kde_est[r],np.log(np.divide(kde_est[r],ffit[r])))*(test_space[1]-test_space[0]))

    return out

from scipy import stats
def DN_Mode(y):
    #y must be numpy array
    if not isinstance(y,np.ndarray):
        y = np.asarray(y)
    return float(stats.mode(y).mode)

import numpy as np
import numba

@numba.jit(nopython=True,parallel=True)
def EN_SampEn(y,M = 2,r = 0,pre = ''):
    if r == 0:
        r = .1*np.std(y)
    # else:
    #     r = r*np.std(y)
    M = M + 1
    N = len(y)
    lastrun = np.zeros(N)
    run = np.zeros(N)
    A = np.zeros(M)
    B = np.zeros(M)
    p = np.zeros(M)
    e = np.zeros(M)

    for i in range(1,N):
        y1 = y[i-1]

        for jj in range(1,N-i + 1):

            j = i + jj - 1

            if np.absolute(y[j] - y1) < r:

                run[jj] = lastrun[jj] + 1
                M1 = min(M,run[jj])
                for m in range(int(M1)):
                    A[m] = A[m] + 1
                    if j < N:
                        B[m] = B[m] + 1
            else:
                run[jj] = 0
        for j in range(N-1):
            lastrun[j] = run[j]

    NN = N * (N - 1) / 2
    p[0] = A[0] / NN
    e[0] = - np.log(p[0])
    for m in range(1,int(M)):
        p[m] = A[m] / B[m-1]
        e[m] = -np.log(p[m])
    i = 0
    out = {'sampen':np.zeros(len(e)),'quadSampEn':np.zeros(len(e))}
    for ent in e:
        quaden1 = ent + np.log(2*r)
        out['sampen'][i] = ent
        out['quadSampEn'][i] = quaden1
        i = i + 1

    return out

from scipy import signal
def SY_Trend(y):
    if not BF_iszscored(y):
        warnings.warn('y should be zero scored. ')
        #y = (y - np.mean(y)) / np.std(y)
    N  = len(y)
    stdRatio = np.std(signal.detrend(y)) / np.std(y)

    gradient, intercept = LinearFit(np.arange(N),y)

    yC = np.cumsum(y)
    meanYC = np.mean(yC)
    stdYC = np.std(yC)

    #print(gradient)
    #print(intercept)

    gradientYC, interceptYC = LinearFit(np.arange(N),yC)

    meanYC12 = np.mean(yC[0:int(np.floor(N/2))])
    meanYC22 = np.mean(yC[int(np.floor(N/2)):])

    out = {'stdRatio':stdRatio,'gradient':gradient,'intercept':intercept,
            'meanYC':meanYC,'stdYC':stdYC,'gradientYC':gradientYC,
            'interceptYC':interceptYC,'meanYC12':meanYC12,'meanYC22':meanYC22}
    return out

def LinearFit(xData,yData):
    m, b = np.polyfit(xData,yData,1)
    return m,b

import numpy as np
import numba

@numba.jit(nopython=True,parallel=True)
def DN_Mean(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(y.mean())

def CO_glscf(y,alpha = 1.0,beta = 1.0,tau = ''):
    if tau == '':
        tau = CO_FirstZero(y,'ac')
    N = len(y)
    beta = float(beta)
    alpha = float(alpha)
    y1 = np.absolute(y[0:N-tau])
    y2 = np.absolute(y[tau:N])
    top = np.mean(np.multiply(np.power(y1,alpha),np.power(y2,beta))) - np.mean(np.power(y1,alpha)) * np.mean(np.power(y2,beta))
    bot =  np.sqrt(np.mean(np.power(y1,2*alpha)) - np.mean(np.power(y1,alpha))**2) * np.sqrt(np.mean(np.power(y2,2*beta)) - np.mean(np.power(y2,beta))**2)
    if bot == 0:
        return np.inf
    glscf = top / bot
    return glscf

def DN_Cumulants(y,cumWhatMay = 'skew1'):
    if cumWhatMay == 'skew1':
        return stats.skew(y)
    elif cumWhatMay == 'skew2':
        return stats.skew(y,0)
    elif cumWhatMay == 'kurt1':
        return stats.kurtosis(y)
    elif cumWhatMay == 'kurt2':
        return stats.kurtosis(y,0)
    else:
         raise Exception('Requested Unknown cumulant must be: skew1, skew2, kurt1, or kurt2')

def DN_Range(y):
    return np.max(y) - np.min(y)

from Periphery import *
def DN_FitKernalSmooth(x,varargin = {}):
    #varargin should be dict with possible keys numcross
    #area and arclength

    out = {}

    m = np.mean(x)

    kde = stats.gaussian_kde(x)
    #i think matlabs kde uses 100 points
    #but end numbers end up being midly off
    #seems to be rounding entropy max, min line up
    test_space = np.linspace(np.min(x),np.max(x),100)

    f = kde(test_space)

    df = np.diff(f)

    ddf  = np.diff(df)

    sdsp = ddf[BF_sgnchange(df,1)]

    out['npeaks'] = sum(sdsp < -.0002)

    out['max'] = np.max(f)

    out['entropy'] = - sum(np.multiply(f[f>0],np.log(f[f>0])))*(test_space[2]-test_space[1])

    out1 = sum(f[test_space > m]) * (test_space[2]-test_space[1])
    out2 = sum(f[test_space < m]) * (test_space[2]-test_space[1])
    out['asym'] = out1 / out2

    out1 = sum(np.absolute(np.diff(f[test_space < m]))) * (test_space[2]-test_space[1])
    out1 = sum(np.absolute(np.diff(f[test_space > m]))) * (test_space[2]-test_space[1])
    out['plsym'] = out1 / out2

    if 'numcross' in varargin:
        thresholds = varargin['numcross']
        out['numCrosses']  = {}
        for i in range(len(thresholds)):
            numCrosses = sum(BF_sgnchange(f - thresholds[i]))
            out['numCrosses'][thresholds[i]] = numCrosses
    if 'area' in varargin:
        thresholds = varargin['area']
        out['area']  = {}
        for i in range(len(thresholds)):
            areaHere = sum(f[f < thresholds[i]]) * (test_space[2]-test_space[1])
            out['area'][thresholds[i]] = areaHere
    if 'arclength' in varargin:
        thresholds = varargin['arclength']
        out['arclength']  = {}
        for i in range(len(thresholds)):
            fd = np.absolute(np.diff(f[(test_space > m - thresholds[i]) & (test_space < m + thresholds[i])]))
            arclengthHere = sum(fd) * (test_space[2]-test_space[1])
            out['arclength'][thresholds[i]] = arclengthHere
    return out

import numpy as np
@numba.jit(nopython=True)
def DN_Median(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.median(y))

#@numba.jit(nopython=True,parallel=True)
def DN_Spread(y,spreadMeasure = 'std'):
    if spreadMeasure == 'std':
        return np.std(y)
    elif spreadMeasure == 'iqr':
        return stats.iqr(y)
    elif spreadMeasure == 'mad':
        return mad(y)
    elif spreadMeasure == 'mead':
        return mead(y)#stats.median_absolute_deviation(y)
    else:
        raise Exception('spreadMeasure must be one of std, iqr, mad or mead')
def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def mead(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)

@numba.jit(nopython=True,parallel=True)
def DN_MinMax(y,which = 'max'):
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    if which == 'min':
        return(y.min())
    else:
        return(y.max())

@numba.jit(nopython=True,parallel=True)
def DN_CustomSkewness(y,whatSkew = 'pearson'):
    if whatSkew == 'pearson':
        if np.std(y) != 0:
            return (3*np.mean(y) - np.median(y)) / np.std(y)
        else:
            return 0
    elif whatSkew == 'bowley':
        qs = np.quantile(y,[.25,.5,.75])
        if np.std(y) != 0:
            return (qs[2] + qs[0] - 2*qs[1]) / (qs[2] - qs[0])
        else:
            return 0

    else:
         raise Exception('whatSkew must be either pearson or bowley.')

def EN_mse(y,scale=range(1,11),m=2,r=.15,adjust_r=True):

    minTSLength = 20
    numscales = len(scale)
    y_cg = []

    for i in range(numscales):
        bufferSize = scale[i]
        y_buffer = BF_makeBuffer(y,bufferSize)
        y_cg.append(np.mean(y_buffer,1))

    outEns = []

    for si in range(numscales):
        if len(y_cg[si]) >= minTSLength:
            sampEnStruct = EN_SampEn(y_cg[si],m,r)
            outEns.append(sampEnStruct)
        else:
            outEns.append(np.nan)
    sampEns = []
    for out in outEns:
        sampEns.append(out['sampen'][1])

    maxSampen = np.max(sampEns)
    maxIndx = np.argmax(sampEns)

    minSampen = np.min(sampEns)
    minIndx = np.argmin(sampEns)

    meanSampen = np.mean(sampEns)

    stdSampen = np.std(sampEns)

    meanchSampen = np.mean(np.diff(sampEns))

    out = {'sampEns':sampEns,'max Samp En':maxSampen,'max point':scale[maxIndx],'min Samp En':minSampen,\
    'min point':scale[minIndx],'mean Samp En':meanSampen,'std Samp En':stdSampen, 'Mean Change':meanchSampen}

    return out

def IN_AutoMutualInfo(y,timeDelay = 1,estMethod = 'gaussian',extraParam = []):
    if isinstance(timeDelay,str):
        timeDelay = CO_FirstZero(y,'ac')
    N = len(y)

    if isinstance(timeDelay,list):
        numTimeDelays = len(timeDelay)
    else:
        numTimeDelays = 1
        timeDelay = [timeDelay]
    amis = []
    out = {}
    for k in range(numTimeDelays):
        y1 = y[0:N-timeDelay[k]]
        y2 = y[timeDelay[k]:N]
        if estMethod == 'gaussian':
            r = np.corrcoef(y1,y2)[1,0]
            amis.append(-.5 * np.log(1 - r**2))
            out['Auto Mutual ' + str(timeDelay[k])] = -.5 * np.log(1 - r**2)

    return out

def EN_CID(y):


    CE1 = f_CE1(y)
    CE2 = f_CE2(y)

    minCE1 = f_CE1(np.sort(y))
    minCE2 = f_CE2(np.sort(y))

    CE1_norm = CE1 / minCE1
    CE2_norm = CE2 / minCE2

    out = {'CE1':CE1,'CE2':CE2,'minCE1':minCE1,'minCE2':minCE2,
            'CE1_norm':CE1_norm,'CE2_norm':CE2_norm}
    return out

def f_CE1(y):
    return np.sqrt(np.mean( 1 + np.power(np.diff(y),2) ) )

def f_CE2(y):
    return np.mean(np.sqrt( 1 + np.power(np.diff(y),2) ) )

def DN_Unique(x):
    return len(np.unique(x)) / len(x)

from scipy import optimize
def DT_IsSeasonal(y):

    N = len(y)

    th_fit = 0.3
    th_ampl = 0.5

    try:
        params, params_covariance = optimize.curve_fit(test_func, np.arange(N), y, p0=[10, 13,600,0])
    except:
        return False

    a,b,c,d = params



    y_pred = a * np.sin(b * np.arange(N) + d) + c

    SST = sum(np.power(y - np.mean(y),2))
    SSr = sum(np.power(y - y_pred,2))

    R = 1 - SSr / SST


    if R > th_fit: #and (np.absolute(a) > th_ampl*.1*np.std(y)):
        return True
    else:
        return False

def test_func(x, a, b,c,d):
    return a * np.sin(b * x + d) + c

#@numba.jit(nopython=True,parallel=True)
def EN_ApEn(y,mnom = 1,rth = .2):

    r = rth * np.std(y)
    N = len(y)
    phi = np.zeros(2)

    for k in range(2):
        m = mnom + k
        m = int(m)
        C = np.zeros(N-m+1)

        x = np.zeros((N - m + 1, m))

        for i in range(N - m + 1):
            x[i,:] = y[i:i+m]

        ax = np.ones((N - m + 1, m))
        for i in range(N-m+1):

            for j in range(m):
                ax[:,j] = x[i,j]

            d = np.absolute(x-ax)
            if m > 1:
                d = np.maximum(d[:,0],d[:,1])
            dr = ( d <= r )
            C[i] = np.sum(dr) / (N-m+1)
        phi[k] = np.mean(np.log(C))
    return phi[0] - phi[1]

import matplotlib.pyplot as plt
def SC_HurstExp(x):

    N = len(x)

    splits = int(np.log2(N))

    rescaledRanges = []

    n = []

    for i in range(splits):

        chunks = 2**(i)

        n.append(int(N / chunks))


        y = x[:N - N % chunks]

        y = y.reshape((chunks,int(N/chunks)))

        m = y.mean(axis = 1,keepdims = True)

        y = y - m

        z = np.cumsum(y,1)

        R = np.max(z,1) - np.min(z,1)

        S = np.std(y,1)

        S[S == 0] = 1


        rescaledRanges.append(np.mean(R/S))

    logRS = np.log(rescaledRanges)
    logn = np.log(n)

    # plt.plot(logn,logRS)
    # plt.show()

    p = np.polyfit(logn,logRS,1)

    return p[0]

def DN_ObsCount(y):
    return np.count_nonzero(~np.isnan(y))

#@numba.jit(nopython=True,parallel=True)
def EN_ShannonEn(y):
    p = np.zeros(len(np.unique(y)))
    n = 0
    for i in np.unique(y):
        p[n] = len(y[y == i]) / len(y)
        n = n + 1

    return -np.sum(p*np.log2(p))

# author: Dominik Krzeminski (dokato)
# https://github.com/dokato/dfa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

# detrended fluctuation analysis

def calc_rms(x, scale):
    """
    windowed Root Mean Square (RMS) with linear detrending.

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    """
    # making an array with data divided in windows
    shape = (x.shape[0]//scale, scale)
    X = np.lib.stride_tricks.as_strided(x,shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coeff = np.polyfit(scale_ax, xcut, 1)
        xfit = np.polyval(coeff, scale_ax)
        # detrending and computing RMS of each window
        rms[e] = np.sqrt(np.mean((xcut-xfit)**2))
    return rms

def dfa(x, scale_lim=[5,9], scale_dens=0.25, show=False):
    """
    Detrended Fluctuation Analysis - measures power law scaling coefficient
    of the given signal *x*.

    More details about the algorithm you can find e.g. here:
    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free
    view on neuronal oscillations, (2012).

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of length 2
        boundaries of the scale, where scale means windows among which RMS
        is calculated. Numbers from list are exponents of 2 to the power
        of X, eg. [5,9] is in fact [2**5, 2**9].
        You can think of it that if your signal is sampled with F_s = 128 Hz,
        then the lowest considered scale would be 2**5/128 = 32/128 = 0.25,
        so 250 ms.
      *scale_dens* = 0.25 : float
        density of scale divisions, eg. for 0.25 we get 2**[5, 5.25, 5.5, ... ]
      *show* = False
        if True it shows matplotlib log-log plot.
    Returns:
    --------
      *scales* : numpy.array
        vector of scales (x axis)
      *fluct* : numpy.array
        fluctuation function values (y axis)
      *alpha* : float
        estimation of DFA exponent
    """
    # cumulative sum of data with substracted offset
    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    for e, sc in enumerate(scales):
        if len(calc_rms(y, sc)**2) == 0:
            continue
        fluct[e] = np.sqrt(np.mean(calc_rms(y, sc)**2))

    # fitting a line to rms data
    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    # if show:
    #     fluctfit = 2**np.polyval(coeff,np.log2(scales))
    #     plt.loglog(scales, fluct, 'bo')
    #     plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
    #     plt.title('DFA')
    #     plt.xlabel(r'$\log_{10}$(time window)')
    #     plt.ylabel(r'$\log_{10}$<F(t)>')
    #     plt.legend()
    #     plt.show()
    #return scales, fluct, coeff[0]
    return coeff[0]

def CO_tc3(y,tau = 'ac'):
    if tau == 'ac':
        tau = CO_FirstZero(y,'ac')
    else:
        tau = CO_FirstMin(y,'mi')
    N = len(y)
    yn = y[0:N-2*tau]
    yn1 = y[tau:N-tau]
    yn2 = y[tau*2:N]
    raw = np.mean(np.multiply(np.multiply(yn,yn1),yn2)) / (np.absolute(np.mean(np.multiply(yn,yn1))) ** (3/2))

    return raw

def DN_nlogL_norm(y):
    muhat, sigmahat = stats.norm.fit(y)
    z = (y - muhat) / sigmahat
    L = -.5*np.power(z,2) - np.log(np.sqrt(2*math.pi)*sigmahat)
    return -sum(L) / len(y) 


def CO_AutoCorr(y,lag = 1,method = 'TimeDomianStat',t=1):
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    if method == 'TimeDomianStat':
        if lag == []:
            acf = [1]
            for i in range(1,len(y)-1):
                acf.append(np.corrcoef(y[:-lag],y[lag:])[0,1])
            return acf
        return(np.corrcoef(y[:-lag],y[lag:])[0,1])
    else:
        N = len(y)
        nFFT = int(2**(np.ceil(np.log2(N)) + 1))
        F = np.fft.fft(y - y.mean(),nFFT)
        F = np.multiply(F,np.conj(F))
        acf = np.fft.ifft(F)
        if acf[0] == 0:
            if lag == []:
                return acf
            return acf[lag]


        acf = acf / acf[0]
        acf = acf.real
        if lag == []:
            return acf
        return acf[lag]

import math
def CO_f1ecac(y):
    N = len(y)
    thresh = 1 / math.exp(1)
    for i in range(1,N):
        auto = CO_AutoCorr(y,i)
        if ( auto - thresh ) < 0:
            return i
    return N

def DN_ProportionValues(x,propWhat = 'positive'):
    N = len(x)
    if propWhat == 'zeros':
        return sum(x == 0) / N
    elif propWhat == 'positive':
        return sum(x > 0) / N
    elif propWhat == 'negative':
        return sum(x < 0) / N
    else:
        raise Exception('Only negative, positve, zeros accepted for propWhat.')

@numba.jit(nopython=True,parallel=True)
def DN_STD(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.std(y))

def CO_trev(y,tau = 'ac'):
        if tau == 'ac':
            tau = CO_FirstZero(y,'ac')
        else:
            tau = CO_FirstMin(y,'mi')
        N = len(y)
        yn = y[0:N-tau]
        yn1 = y[tau:N]
        
        raw = np.mean(np.power(yn1-yn,3)) / np.mean(np.power(yn1 - yn,2))**(3/2)

        return raw

import warnings
@numba.jit(nopython=True,parallel=True)
def DN_cv(x,k = 1):
    # if k % 1 != 0 or k < 0:
    #     warnings.warn("k should probably be positive int")
    return (np.std(x)**k) / (np.mean(x)**k)

#@numba.jit(nopython=True,parallel=True)
def DN_TrimmedMean(y,n = 0):
    N = len(y)
    trim = int(np.round(N * n / 2))
    y = np.sort(y)
    #return stats.trim_mean(y,n) doesn't agree with matlab
    return np.mean(y[trim:N-trim])

def SC_DFA(y):

    N = len(y)

    tau = int(np.floor(N/2))

    y = y - np.mean(y)

    x = np.cumsum(y)

    taus = np.arange(5,tau+1)

    ntau = len(taus)

    F = np.zeros(ntau)

    for i in range(ntau):

        t = int(taus[i])



        x_buff = x[:N - N % t]

        x_buff = x_buff.reshape((int(N / t),t))


        y_buff = np.zeros((int(N / t),t))

        for j in range(int(N / t)):

            tt = range(0,int(t))

            p = np.polyfit(tt,x_buff[j,:],1)

            y_buff[j,:] =  np.power(x_buff[j,:] - np.polyval(p,tt),2)



        y_buff.reshape((N - N % t,1))

        F[i] = np.sqrt(np.mean(y_buff))

    logtaur = np.log(taus)

    logF = np.log(F)

    p = np.polyfit(logtaur,logF,1)

    return p[0]

@numba.jit(nopython=True,parallel=True)
def DN_HighLowMu(y):
    mu = np.mean(y)
    mhi = np.mean(y[y>mu])
    mlo = np.mean(y[y<mu])
    return (mhi - mu) / (mu - mlo)

