import numpy as np
def EN_SampEn(y,M = 2,r = 'empty',pre = ''):
    if r == 'empty':
        r = .1*np.std(y)
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
    out = {}
    for ent in e:
        quaden1 = ent + np.log(2*r)
        out['sampen' + str(i)] = ent
        out['quadSampEn' + str(i)] = quaden1
        i = i + 1

    return out
