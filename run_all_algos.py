import numpy as np
import multiprocessing as mp
from functools import partial
import pandas as pd
import make_operations
operations = make_operations.make_operations()
make_operations.make_otherfunctions()
from Operations import *
from Periphery import *

def read_in_data(id):
    id = str(id).zfill(4)
    data = np.genfromtxt('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/HR/UVA' + id +'_hr.csv', delimiter=',')
    time = data[:,0]
    hr = data[:,1]
    return(time,hr)




def run_histogram_algos(y,algos = 'all',results = {},impute = False):

    if impute:
        y = impute(y)
    else:
        y = y[~np.isnan(y)]

    if 'DN_Mean' in algos:
        results['mean'] = DN_Mean(y)

    if 'DN_Range' in algos:
        results['range'] = DN_Range(y)

    if 'DN_IQR' in algos:
        results['iqr'] = DN_IQR(y)

    if 'DN_Median' in algos:
        results['median'] = DN_Median(y)

    if 'DN_MinMax' in algos:
        results['max'] = DN_MinMax(y)
        results['min'] = DN_MinMax(y,'min')

    if 'DN_Mode' in algos:
        results['mode'] = DN_Mode(y)

    if 'DN_Cumulants' in algos:
        results['skew1'] = DN_Cumulants(y,'skew1')
        results['skew2'] = DN_Cumulants(y,'skew2')
        results['kurt1'] = DN_Cumulants(y,'kurt1')
        results['kurt2'] = DN_Cumulants(y,'kurt2')

    if 'DN_Burstiness' in algos:
        results['Burstiness'] = DN_Burstiness(y)

    if 'DN_Unique' in algos:
        results['Percent Unique'] = DN_Unique(y)

    if 'DN_Withinp' in algos:
        results['Within 1 std'] = DN_Withinp(y)
        results['Within 2 std'] = DN_Withinp(y,2)

    if 'EN_ShannonEn':
        results['Shannon Entropy'] = EN_ShannonEn(y)

    if 'DN_STD' in algos:
        results['std'] = DN_STD(y)
        if results['std'] == 0:
            return results

    if 'DN_Moments' in algos:
        results['Moment 2'] = DN_Moments(y,2)
        results['Moment 3'] = DN_Moments(y,3)
        results['Moment 4'] = DN_Moments(y,4)
        results['Moment 5'] = DN_Moments(y,5)
        results['Moment 6'] = DN_Moments(y,6)

    if 'DN_pleft' in algos:
        results['pleft'] = DN_pleft(y)

    if 'DN_CustomSkewness' in algos:
        results['Pearson Skew'] = DN_CustomSkewness(y)

    if 'DN_HighLowMu' in algos:
        results['High Low Mean Ratio'] = DN_HighLowMu(y)


    if 'DN_nlogL_norm' in algos:
        results['Log liklihood of Norm fit'] = DN_nlogL_norm(y)

    if 'DN_Quantile' in algos:
        results['Quantile 50'] = DN_Quantile(y)
        results['Quantile 75'] = DN_Quantile(y,.75)
        results['Quantile 90'] = DN_Quantile(y,.90)
        results['Quantile 95'] = DN_Quantile(y,.95)
        results['Quantile 99'] = DN_Quantile(y,.99)

    if 'DN_RemovePoints' in algos:
        out = DN_RemovePoints(y,p = .5)
        results = parse_outputs(out,results,'DN_RemovePoints')

    if 'DN_Spread':
        results['Mean Abs Deviation'] = DN_Spread(y,'mad')
        results['Median Abs Deviation'] = DN_Spread(y,'mead')

    if 'DN_TrimmedMean' in algos:
        results['trimmed mean 50'] = DN_TrimmedMean(y,.5)
        results['trimmed mean 75'] = DN_TrimmedMean(y,.75)
        results['trimmed mean 25'] = DN_TrimmedMean(y,.25)

    if 'DN_cv' in algos:
        results['DN_cv 1'] = DN_cv(y)
        results['DN_cv 2'] = DN_cv(y,2)
        results['DN_cv 3'] = DN_cv(y,3)

    return results

def time_series_dependent_algos(y,algos,results,t):

        #print('Corr')
        if 'CO_AutoCorr' in algos:
            corr = CO_AutoCorr(y,[],'Forier',t)

            i = 0

            for c in corr:
                if i > 25:
                    break
                elif i == 0:
                    i = i + 1
                    continue

                results['AutoCorr lag ' + str(i)] = c
                i = i + 1

        #print('f1')
        if 'CO_f1ecac' in algos:
            results['f1ecac'] = CO_f1ecac(y)

        #print('first min')
        if 'CO_FirstMin' in algos:
            results['FirstMin'] = CO_FirstMin(y)

        if 'CO_FirstZero' in algos:
            results['FirstZero'] = CO_FirstZero(y)

        #print('glscf')
        if 'CO_glscf' in algos:
            for alpha in range(1,5):
                for beta in range(1,5):
                    results['glscf ' + str(alpha) + ' ' + str(beta)] = CO_glscf(y,alpha,beta)

        if 'CO_tc3' in algos:
            results['tc3'] = CO_tc3(y)

        if 'CO_trev' in algos:
            results['trev'] = CO_trev(y)

        # if 'dfa' in algos:
        #     results['dfa'] = dfa(y)

        if 'DN_CompareKSFit' in algos:
            out = DN_CompareKSFit(y)
            results = parse_outputs(out,results,'DN_CompareKSFit')


        if 'DT_IsSeasonal' in algos:
            results['IsSeasonal?'] = DT_IsSeasonal(y)

        if 'EN_ApEn' in algos:
            results['ApEn'] = EN_ApEn(y)

        if 'EN_CID' in algos:
            out = EN_CID(y)
            results = parse_outputs(out,results,'Complexity')

        if 'EN_PermEn' in algos:
            results['PermEn 2, 1'] = EN_PermEn(y)
            results['PermEn 3, 6'] = EN_PermEn(y,3,6)

        if 'EN_SampEn' in algos:
            out = EN_SampEn(y)
            results = parse_outputs(out,results,'Sample Entropy 2, .1')
            out = EN_SampEn(y,3,.25)
            results = parse_outputs(out,results,'Sample Entropy 3, .25')

        if 'IN_AutoMutualInfo' in algos:
            out = IN_AutoMutualInfo(y)
            results = parse_outputs(out,results,'Auto Mutual Info')

        if 'SY_Trend'in algos:
            if not BF_iszscored(y):
                out = SY_Trend((y-np.mean(y)) / np.std(y))
            else:
                out = SY_Trend(y)
            results = parse_outputs(out,results,'Trend')

        if 'SC_HurstExp' in algos:
            results['Hurst Exp'] = SC_HurstExp(y)
        if 'SC_DFA' in algos:
            results['DFA alpha'] = SC_DFA(y)

        return results

def run_algos(y,algos = 'all',last_non_nan = np.nan,t=1):

    results = {}

    if algos == 'all':
        algos = operations

    #Make sure time series isn't empty
    if 'DN_ObsCount' in algos:
        results['Observations'] = DN_ObsCount(y)
        if results['Observations'] == 0:
            return results

    #Compute all histogram stats on non-imputed data
    results = run_histogram_algos(y,algos,results)

    #if y is only 1 value don't calc time depedent stuff
    if results['std'] == 0.0:

        return results

    #impute data for algos that can't run with nans
    y = impute(y,last_non_nan)

    results = time_series_dependent_algos(y,algos,results,t)

    return results

def parse_outputs(outputs,results,func):
    for key in outputs:
        if isinstance(outputs[key],list) or isinstance(outputs[key],np.ndarray):
            i = 1
            for out in outputs[key]:
                results[func + ' ' + key + ' ' + str(i)] = out
                i = i + 1
        else:
            results[func + ' ' + key] = outputs[key]
    return results




def get_interval(interval_length,end_time,times):
    # interval_length is in seconds
    # end_time is in seconds
    return np.where((times <= end_time) & (times > end_time - interval_length))

def all(t,hr = hr,time = time,interval_length = 60*10):
    indx = get_interval(interval_length,int(t),time)
    results = run_algos(hr[indx])
    results['time'] = t
    return results

def all_times(t,series,time,interval_length = 600):
    indx = get_interval(interval_length,int(t),time)
    indx = indx[0]
    if series[np.min(indx)] == np.nan:
        nonnan = np.argwhere(~np.isnan(series))[np.argwhere(~np.isnan(series)) < np.min(indx)]
        if len(nonnan) != 0:
            last_non_nan_indx = np.max(nonnan)
            lastvalue = series[last_non_nan_indx]
        else:
            lastvalue = np.nan

        results = run_algos(series[indx],'all',lastvalue,t)
    else:

        results = run_algos(series[indx],'all',1,t)
    results['time'] = t
    return results

def impute(y,last):
    if y[0] == np.nan and last == np.nan:
        min = np.min(np.argwhere(~np.isnan(y)))
        return impute(y[min:],last)
    elif y[0] == np.nan:
        y[0] = last
    y = nanfill(y)
    return y

def nanfill(y):
    for i in np.argwhere(np.isnan(y)):
        y[i] = y[i-1]
    return y

def run_all(time_series,time,interval_length = 60*10,step_size=60*5):
    end_times = np.arange(np.min(time) + interval_length,np.max(time),step_size)
    if not isinstance(time_series,dict):
        time_series = {'y':time_series}
    full_results = {}
    for key, data in time_series.items():
        print("Starting to Analyze " + key)
        pool = mp.Pool(mp.cpu_count())
        #results = [pool.apply(all, args=(hr,time,interval_length,t)) for t in end_times]
        results = pool.map(partial(all_times,series = data,time = time), [t for t in end_times])
        pool.close()
        #results = all_times(end_times[1],data,time)
        first = True
        for result in results:
            if first:

                df = pd.DataFrame(result,index = [0])[:1]
                first  = False
                continue
            else:
                df = df.append(results, ignore_index=True)
        full_results[key] = df
        #break
    return full_results




# algos = 'all'
# interval_length = 60*10
# step_size = 60*5
# for i in range(211,212):
#     print(i)
#     time, hr = read_in_data(i)
#     id = str(i).zfill(4)
#     end_times = np.arange(np.min(time) + interval_length,np.max(time),step_size)
#
#
#
#     pool = mp.Pool(mp.cpu_count())
#     #results = [pool.apply(all, args=(hr,time,interval_length,t)) for t in end_times]
#     results = pool.map(all, [t for t in end_times])
#
#     pool.close()
#     first = True
#     for result in results:
#         if first:
#
#             df = pd.DataFrame(result,index = [0])[:1]
#             first  = False
#             continue
#         df = df.append(results, ignore_index=True)
#     df.to_csv('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/hctsa results/UVA' + id +'_hist_test2.csv',index = False)
# # # for t in end_times:
#     print(t)
#     indx = get_interval(interval_length,int(t),time)
#     results = run_algos(hr[indx])
#     results['time'] = t
#     if first:
#         df = pd.DataFrame(results,index = [0])[:1]
#         first = False
#         continue
#     df = df.append(results, ignore_index=True)
#     df.to_csv('test.csv',index = False)
