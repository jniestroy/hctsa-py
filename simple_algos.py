import statistics as stats
from scipy import stats
import warnings

import math
import scipy.io
import time
import numpy as np
start_time = time.time()
import make_operations
make_operations.make_operations()
make_operations.make_otherfunctions()
from Operations import *
from Periphery import *

functions = [CO_AutoCorr,CO_f1ecac,CO_FirstMin,CO_FirstZero]
functions = [DN_ObsCount]
mat = scipy.io.loadmat('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/RR/UVA0013_rr.mat',squeeze_me=True)
all = []
rr = np.asarray(mat['rr'])
rr = rr[rr < 2000]
#rr = rr[0:500]
m = []
y = np.ones(5000)
y = np.asarray([4,7,9,10,6,11,3])
#x  = 100 * np.sin(5 * y) + 25 + np.random.normal(0,5,5000)
start_time = time.time()
#for func in functions:
    #m.append(CO_AutoCorr(rr,i,'test'))
for i in range(1):
    m.append(SC_HurstExp(rr[:2000]))
print(m)
print("--- %s seconds ---" % (time.time() - start_time))


# functions = [
# for i in range(1,2800):
#     id = str(i).zfill(4)
#     mat = scipy.io.loadmat('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/RR/UVA' + id +'_rr.mat',squeeze_me=True)
#     rr = np.asarray(mat['rr'])
#     m = []
#     for func in functions:
#         m.append(func(rr))
#     all.append(m)
# print(all)
# print("--- %s seconds ---" % (time.time() - start_time))

# def DN_Mean(y):
#     return(y.mean())
# # mat = scipy.io.loadmat('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/RR/UVA0003_rr.mat',squeeze_me=True)
# # print(mat)
# start_time = time.time()
# m = []
# total = 0
# for i in range(1,301):
#     id = str(i).zfill(4)
#
#     mat = scipy.io.loadmat('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/RR/UVA' + id +'_rr.mat',squeeze_me=True)
#
#     rr = np.asarray(mat['rr'])
#     test = time.time()
#     m.append(DN_Mean(rr))
#     total = total + (time.time() - test)
# print("--- %s seconds ---" % (time.time() - start_time))
# print('Reading in Files took: ' + str(total))
