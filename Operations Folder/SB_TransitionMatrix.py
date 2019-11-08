import scipy
def SB_TransitionMatrix(y,howtocg = 'quantile',numGroups = 2,tau = 1):

    if tau == 'ac':
        tau = CO_FirstZero(y,'ac')

    if tau > 1:
        y = scipy.signal.resample(y,math.ceil(len(y) / tau))

    N = len(y)

    yth = SB_CourseGrain(y,howtocg,numGroups)

    if yth.shape[1] > yth.shape[0]:
        yth = yth.transpose()

    T = np.zeros(numGroups)
