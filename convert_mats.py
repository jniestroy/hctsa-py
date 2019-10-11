import numpy as np
import scipy.io
for i in range(1,230):
    id  = str(i).zfill(4)
    mat = scipy.io.loadmat('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/RR/UVA' + id +'_rr.mat',squeeze_me=True)
    all = []
    rr = np.asarray(mat['rr'])
    rr = rr[rr < 2000] / 1000
    time = np.cumsum(rr)
    hr = (1 / rr) * 60
    with open('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/HR/UVA' + id +'_hr.csv','w') as f:
        for x in range(len(hr)):
            line = str(time[x]) + ',' + str(hr[x])  + '\n'
            f.write(line)
