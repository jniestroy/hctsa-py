import statistics as stats
from scipy import stats
import warnings
import h5py
import math
import scipy.io
import time
import numpy as np
import pandas as pd
start_time = time.time()
import make_operations
make_operations.make_operations()
make_operations.make_otherfunctions()
from Operations import *
from Periphery import *

def get_column_names(vname,f):
    names = []
    for name in vname[0]:
        obj = f[name]
        col_name = ''.join(chr(i) for i in obj[:])
        names.append(col_name)
    return names

def read_in_NICU_file(path):
    arrays = {}
    f = h5py.File(path,'r')
    for k, v in f.items():
        arrays[k] = np.array(v)

    df = pd.DataFrame(np.transpose(arrays['vdata']),columns = get_column_names(f['vname'],f))
    times = pd.Series(arrays['vt'][0], index=df.index)

    return df,times


functions = [CO_AutoCorr,CO_f1ecac,CO_FirstMin,CO_FirstZero,]
functions = [DN_ObsCount]
mat = scipy.io.loadmat('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/RR/UVA0013_rr.mat',squeeze_me=True)
all = []
rr = np.asarray(mat['rr'])
rr = rr[rr < 2000]
filepath = '/Users/justinniestroy-admin/Desktop/NICU Vitals/UVA_6738_vitals.mat'
# df,time = read_in_NICU_file(filepath)
# time = time.to_numpy()
# num_cols = df.shape[1]
# time_series = {}
# for i in range(num_cols):
#     time_series[list(df.columns.values)[i]] = df[list(df.columns.values)[i]].to_numpy()
#rr = rr[0:500]
functions = [CO_RM_AMInformation]
#functions  = [DN_Moments,DN_Withinp,DN_Quantile,FC_Suprise,MD_hrv_classic,MD_pNN,EN_PermEn, DN_OutlierInclude,DN_Burstiness,DN_pleft,CO_FirstZero,DN_Fit_mle,CO_FirstMin,DN_IQR,DN_CompareKSFit,DN_Mode,EN_SampEn,SY_Trend,DN_Mean,CO_glscf,DN_Cumulants,DN_Range,DN_FitKernalSmooth,DN_Median,DN_Spread,DN_MinMax,DN_CustomSkewness,EN_mse,IN_AutoMutualInfo,EN_CID,DN_Unique,DT_IsSeasonal,EN_ApEn,SC_HurstExp,DN_ObsCount,EN_ShannonEn,CO_tc3,DN_nlogL_norm,CO_AutoCorr,CO_f1ecac,DN_ProportionValues,DN_STD,CO_trev,DN_cv,DN_TrimmedMean,SC_DFA,DN_HighLowMu]

m = []
y = np.ones(5000)
y = np.asarray([141,142,141,145,151,153,158,141,141,142])
# y = np.random.rand(500)
# x = np.random.rand(500)
# z = np.random.rand(1000)
#x  = 100 * np.sin(5 * y) + 25 + np.random.normal(0,5,5000)
#start_time = time.time()
for func in functions:
    start_time = time.time()
    print(func(y,5))

#
# min = 500
# test = {}
# for func in functions:
#     start_time = time.time()
#     func(y)
#     func(x)
#     func(z)
#     took = time.time() - start_time
#     test[str(func).split(' ')[1]] = took
#     if took < min:
#         min = took
# test['min']= min
# import json
# with open('result-new.json', 'w') as fp:
#     json.dump(test, fp)
#
# for key in test:
#     test[key] = test[key] / min
# print(test)



# print(len(time_series['HR']))
# for i in range(1,int(58000 / 300)):
#     print(i)
#     m.append(EN_SampEn(time_series['HR'][(i-1)*300:i*300],3,.25))
# print(m)
#print("--- %s seconds ---" % (time.time() - start_time))


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
