import numpy as np
import multiprocessing as mp
from functools import partial
import pandas as pd
import csv
#import make_operations
#operations = make_operations.make_operations()
#make_operations.make_otherfunctions()
from Operations import *
from Periphery import *
import time

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
        if np.count_nonzero(np.isnan(y)) > 0:
            y = impute(y,np.nan)
            #raise Exception('Missing Value')
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
            results['Sample Entropy'] = out["Sample Entropy"]
            results["Quadratic Entropy"] = out["Quadratic Entropy"]


        if 'IN_AutoMutualInfo' in algos:
            out = IN_AutoMutualInfo(y)
            results = parse_outputs(out,results,'Auto Mutual Info')

        if 'SY_Trend'in algos:
            if not BF_iszscored(y):
                out = SY_Trend((y-np.mean(y)) / np.std(y))
            else:
                out = SY_Trend(y)
            results = parse_outputs(out,results,'Trend')

        # if 'SC_HurstExp' in algos:
        #     results['Hurst Exp'] = SC_HurstExp(y)
        # if 'SC_DFA' in algos:
        #     results['DFA alpha'] = SC_DFA(y)

        return results

def round2(y,results = {}):
    #sresults = {}
    if np.count_nonzero(np.isnan(y)) > 0:
        y = impute(y,np.nan)
    start = time.time()
    out  = FC_Suprise(y)
    results = parse_outputs(out,results,'FC_Suprise')
    results['FC_Suprise Time'] = time.time() - start
    start = time.time()
    for i in range(2,4):
        for j in range(2,6):
            out = EN_PermEn(y,i,j)
            if isinstance(out,dict):
                results = parse_outputs(out,results,'EN_PermEn '+ str(i) + ' ,'  + str(j))
    results['EN_PermEm Time'] = time.time() - start
    start = time.time()
    for i in range(3,5):
        for j in [.15,.3]:
            out = EN_SampEn(y,i,j)
            results['Sample Entropy ' + str(i) + ' ' + str(j)] = out["Sample Entropy"]
            results["Quadratic Entropy "+ str(i) + ' ' + str(j)] = out["Quadratic Entropy"]
    results['EN_SampEn Time'] = time.time() - start
    start = time.time()
    try:
        out = MD_hrv_classic(y)
        results = parse_outputs(out,results,'MD_hrv')
    except:
        print('Failed hrv')
    out = MD_pNN(y)
    results = parse_outputs(out,results,'MD_pNN')

    results['SC_HurstExp'] = SC_HurstExp(y)
    results['Med Time'] = time.time() - start
    # results['SC_DFA'] = SC_DFA(y)
    start = time.time()
    for i in range(2,4):
        for j in [.2]:

            out = EN_mse(y,range(2,8),i,j)

            results = parse_outputs(out,results,'EN_mse '+ str(i) + ' ,'  + str(j))
    results['EN_mse Time'] = time.time() - start

    start = time.time()
    for n in [10,20,50,100,250]:
        out = SY_LocalGlobal(y,'l',n)
        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_LocalGlobal_l' + str(n))

    for n in [.05,.1,.2,.5]:
        out = SY_LocalGlobal(y,'p',n)
        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_LocalGlobal_p' + str(n))

    for n in [10,20,50,100,250]:
        out = SY_LocalGlobal(y,'unicg',n)
        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_LocalGlobal_unicg' + str(n))

    for n in [10,20,50,100,250]:
        out = SY_LocalGlobal(y,'randcg',n)
        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_LocalGlobal_randcg' + str(n))
    results['SY_LocalGlobal Time'] = time.time() - start
    start = time.time()
    for i in range(0,16):
        try:
            results['CO_RM_AMInformation ' + str(i)] = CO_RM_AMInformation(y,i)
        except:
            continue
    results['CO_RM_AMInformation Time']= time.time() - start
    start = time.time()
    for i in range(2,6):

        for tau in range(1,5):

            out = SB_TransitionMatrix(y,'quantile',i,tau)
            results = parse_outputs(out,results,'SB_TransitionMatrix' + str(i) + str(tau))
    results['SB_TransitionMatrix Time'] = time.time() - start

    start = time.time()
    for i in [25,50,100,150,200]:

        out = SY_SpreadRandomLocal(y,i)

        if isinstance(out,dict):
            results = parse_outputs(out,results,'SY_SpreadRandomLocal' + str(i))

    results['SY_SpreadRandomLocal Time'] = time.time() - start
    start = time.time()
    for l in ['l','n']:

        for n in [25,50,75,100]:

            out = ST_LocalExtrema(y,l,n)
            if isinstance(out,dict):
                results = parse_outputs(out,results,'ST_LocalExtrema_' + l + str(n))
    results['ST_LocalExtrema Time'] = time.time() - start

    start = time.time()
    for prop in ['biasprop','momentum','runningvar','prop']:
        if prop == 'prop':
            parameters = [[.1],[.5],[.9]]
        elif prop == 'biasprop':
            parameters = [[.5,.1],[.1,.5]]
        elif prop == 'momentum':
            parameters = [[2],[5],[10]]
        elif prop == 'runningvar':
            parameters = [[]]
        for para in parameters:
            out = PH_Walker(y,prop,para)
            if isinstance(out,dict):
                results = parse_outputs(out,results,'PH_Walker' + str(prop) + str(parameters))
    results['PH_Walker Time'] = time.time() - start
    out = SY_PeriodVital(y)
    results = parse_outputs(out,results,'SY_PeriodVital')

    return results

def run_algos(y,algos = 'all',last_non_nan = np.nan,t=1):

    results = {}

    if algos == 'all':
        algos = ['EN_PermEm', 'DN_Moments', 'DN_Withinp', 'DN_Quantile', 'DN_RemovePoints', 'DN_OutlierInclude', 'DN_Burstiness', 'DN_pleft', 'CO_FirstZero', 'DN_Fit_mle', 'CO_FirstMin', 'DN_IQR', 'DN_CompareKSFit', 'DN_Mode', 'EN_SampEn', 'SY_Trend', 'DN_Mean', 'CO_glscf', 'DN_Cumulants', 'DN_Range', 'DN_FitKernalSmooth', 'DN_Median', 'DN_Spread', 'DN_MinMax', 'DN_CustomSkewness', 'EN_mse', 'IN_AutoMutualInfo', 'EN_CID', 'DN_Unique', 'DT_IsSeasonal', 'EN_ApEn', 'SC_HurstExp', 'DN_ObsCount',  'EN_ShannonEn', 'dfa', 'CO_tc3', 'DN_nlogL_norm', 'CO_AutoCorr', 'CO_f1ecac', 'DN_ProportionValues', 'DN_STD', 'CO_trev', 'DN_cv', 'DN_TrimmedMean', 'SC_DFA', 'DN_HighLowMu']

    if algos == '2':
        algos = ['EN_PermEm', 'FC_Suprise','MD_hrv_classic','MD_pNN','DN_Moments', 'DN_Withinp', 'DN_Quantile', 'DN_RemovePoints', 'DN_OutlierInclude', 'DN_Burstiness', 'DN_pleft', 'CO_FirstZero', 'DN_Fit_mle', 'CO_FirstMin', 'DN_IQR', 'DN_CompareKSFit', 'DN_Mode', 'EN_SampEn', 'SY_Trend', 'DN_Mean', 'CO_glscf', 'DN_Cumulants', 'DN_Range', 'DN_FitKernalSmooth', 'DN_Median', 'DN_Spread', 'DN_MinMax', 'DN_CustomSkewness', 'EN_mse', 'IN_AutoMutualInfo', 'EN_CID', 'DN_Unique', 'DT_IsSeasonal', 'EN_ApEn', 'SC_HurstExp', 'DN_ObsCount',  'EN_ShannonEn', 'dfa', 'CO_tc3', 'DN_nlogL_norm', 'CO_AutoCorr', 'CO_f1ecac', 'DN_ProportionValues', 'DN_STD', 'CO_trev', 'DN_cv', 'DN_TrimmedMean', 'SC_DFA', 'DN_HighLowMu']

        if DN_ObsCount(y) < 10:
            return results

        y_imputed = impute(y,last_non_nan)

        results = round2(y_imputed)

        return(results)
    #Make sure time series isn't empty
    if 'DN_ObsCount' in algos:
        results['Observations'] = DN_ObsCount(y)
        if results['Observations'] <= 10:
            return results
    if len(algos)>1:
    #Compute all histogram stats on non-imputed data
        start = time.time()
        results = run_histogram_algos(y,algos,results)
        results['Hist Time'] = time.time() - start
    else:
        return results
    #if y is only 1 value don't calc time depedent stuff
    if results['std'] == 0.0:

        return results

    #impute data for algos that can't run with nans
    start = time.time()
    y_imputed = impute(y,last_non_nan)
    results['Impute Time'] = time.time() - start

    start = time.time()
    results = time_series_dependent_algos(impute(y,last_non_nan),algos,results,t)
    results['Time1 Time'] = time.time() - start
    start = time.time()
    results = round2(y_imputed,results)
    results['Round2 Time'] = time.time() - start
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

def all(t,hr,time,interval_length = 60*10):
    indx = get_interval(interval_length,int(t),time)
    results = run_algos(hr[indx])
    results['time'] = t
    return results

def all_times(t,series,time,interval_length = 600,algos = 'all'):
    indx = get_interval(interval_length,int(t),time)
    indx = indx[0]
    if len(indx) <= 1:
        return {'time':t}
    if np.isnan(series[np.min(indx)]):
        nonnan = np.argwhere(~np.isnan(series))[np.argwhere(~np.isnan(series)) < np.min(indx)]
        if len(nonnan) != 0:
            last_non_nan_indx = np.max(nonnan)
            lastvalue = series[last_non_nan_indx]
        else:
            lastvalue = np.nan
        results = run_algos(series[indx],algos,lastvalue,t)
        #results = run_algos(series[indx],['DN_ObsCount'],lastvalue,t)
    else:
        #results = run_algos(series[indx],['DN_ObsCount'],1,t)
        results = run_algos(series[indx],algos,1,t)
    results['time'] = t
    return results


def impute(y_test,last):
    if np.isnan(y_test[0]) and np.isnan(last):
        min = np.min(np.argwhere(~np.isnan(y_test)))
        return y_test[min:]
    elif np.isnan(y_test[0]):
        y_test[0] = last
    y_test = nanfill(y_test)
    return y_test

def nanfill(x):
    for i in np.argwhere(np.isnan(x)):
        x[i] = x[i-1]
    return x

def run_all(time_series,time1,id,interval_length = 60*10,step_size=60*10,algos = 'all'):
    end_times = np.arange(np.min(time1) + interval_length,np.max(time1),step_size)
    if not isinstance(time_series,dict):
        time_series = {'y':time_series}
    full_results = {}
    for key, data in time_series.items():
        print("Analyzing " + key)

        np.seterr(divide='ignore')
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        #results = [pool.apply(all, args=(hr,time,interval_length,t)) for t in end_times]
        results = pool.map(partial(all_times,series = data,time = time1,algos = algos), [t for t in end_times])
        pool.close()

        print("Performing Calcs took " + str(time.time() - start))
        #results = all_times(end_times[1],data,time)

        if algos == '2':
            print(len(results[15].keys()))
            for guy in results:
                if len(guy) > 867:
                    columns = list(guy.keys())
                    break

            for result in results:
                for column in columns:
                    if column not in result.keys():
                        result[column] = ''

            with open('/Results/UVA_' + str(id)  + key + '.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(results)

            return

        columns = ['Observations','mean','range','iqr','median', 'max', 'min', 'mode', 'skew1', 'skew2', 'kurt1', 'kurt2', 'Burstiness', 'Percent Unique', 'Within 1 std', 'Within 2 std', 'Shannon Entropy', 'std', 'Moment 2', 'Moment 3', 'Moment 4', 'Moment 5', 'Moment 6', 'pleft', 'Pearson Skew', 'High Low Mean Ratio', 'Log liklihood of Norm fit', 'Quantile 50', 'Quantile 75', 'Quantile 90', 'Quantile 95', 'Quantile 99', 'DN_RemovePoints fzcacrat', 'DN_RemovePoints ac1rat', 'DN_RemovePoints ac1diff', 'DN_RemovePoints ac2rat', 'DN_RemovePoints ac2diff', 'DN_RemovePoints ac3rat', 'DN_RemovePoints ac3diff', 'DN_RemovePoints sumabsacfdiff', 'DN_RemovePoints mean', 'DN_RemovePoints median', 'DN_RemovePoints std', 'DN_RemovePoints skewnessrat', 'DN_RemovePoints kurtosisrat', 'Mean Abs Deviation', 'Median Abs Deviation', 'trimmed mean 50', 'trimmed mean 75', 'trimmed mean 25', 'DN_cv 1', 'DN_cv 2', 'DN_cv 3', 'AutoCorr lag 1', 'AutoCorr lag 2', 'AutoCorr lag 3', 'AutoCorr lag 4', 'AutoCorr lag 5', 'AutoCorr lag 6', 'AutoCorr lag 7', 'AutoCorr lag 8', 'AutoCorr lag 9', 'AutoCorr lag 10', 'AutoCorr lag 11', 'AutoCorr lag 12', 'AutoCorr lag 13', 'AutoCorr lag 14', 'AutoCorr lag 15', 'AutoCorr lag 16', 'AutoCorr lag 17', 'AutoCorr lag 18', 'AutoCorr lag 19', 'AutoCorr lag 20', 'AutoCorr lag 21', 'AutoCorr lag 22', 'AutoCorr lag 23', 'AutoCorr lag 24', 'AutoCorr lag 25', 'f1ecac', 'FirstMin', 'FirstZero', 'glscf 1 1', 'glscf 1 2', 'glscf 1 3', 'glscf 1 4', 'glscf 2 1', 'glscf 2 2', 'glscf 2 3', 'glscf 2 4', 'glscf 3 1', 'glscf 3 2', 'glscf 3 3', 'glscf 3 4', 'glscf 4 1', 'glscf 4 2', 'glscf 4 3', 'glscf 4 4', 'tc3', 'trev', 'DN_CompareKSFit adiff', 'DN_CompareKSFit peaksepy', 'DN_CompareKSFit relent', 'IsSeasonal?', 'ApEn', 'Complexity CE1', 'Complexity CE2', 'Complexity minCE1', 'Complexity minCE2', 'Complexity CE1_norm', 'Complexity CE2_norm',  'Sample Entropy',  'Quadratic Entropy', 'Auto Mutual Info Auto Mutual 1', 'Trend stdRatio', 'Trend gradient', 'Trend intercept', 'Trend meanYC', 'Trend stdYC', 'Trend gradientYC', 'Trend interceptYC', 'Trend meanYC12', 'Trend meanYC22', 'Hurst Exp', 'DFA alpha', 'time']
        max = 0
        start = time.time()
        for guy in results:
            if len(guy) > max:
                columns = list(guy.keys())
                max = len(guy)
        #print('Loop through results took ' + str(time.time() - start))
        #print('Number of outputs is: ' + str(len(columns)))
        for result in results:
            for column in columns:
                if column not in result.keys():
                    result[column] = ''

        with open('/Results/UVA_' + str(id) + '_' + key + '_less_samp.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

    return full_results
