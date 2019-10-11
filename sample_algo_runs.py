import h5py
import numpy as np
#import multiprocessing as mp
import pandas as pd
import make_operations
operations = make_operations.make_operations()
make_operations.make_otherfunctions()
from Operations import *
from Periphery import *
import run_all_algos as run

#Reads in houlter data
def read_in_data(id):
    id = str(id).zfill(4)
    data = np.genfromtxt('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/HR/UVA' + id +'_hr.csv', delimiter=',')
    time = data[:,0]
    hr = data[:,1]
    return(time,hr)


#Reads in new nicu data from Doug
def read_in_NICU_file(path):
    arrays = {}
    f = h5py.File(path,'r')
    for k, v in f.items():
        arrays[k] = np.array(v)

    df = pd.DataFrame(np.transpose(arrays['vdata']),columns = get_column_names(f['vname'],f))
    times = pd.Series(arrays['vt'][0], index=df.index)

    return df,times

def get_column_names(vname,f):
    names = []
    for name in vname[0]:
        obj = f[name]
        col_name = ''.join(chr(i) for i in obj[:])
        names.append(col_name)
    return names

########################
#
#Houlter Data way to run
#
########################
id = 7
time, hr = read_in_data(id)

time_series = {'hr':hr}
interval_length = 60*10
step_size = 60*5
result = run.run_all(time_series,time)
for series in result:
    test = result[series].head()
    test.to_csv('/Users/justinniestroy-admin/Desktop/UVA_' + str(id) + '_' + series + '_all.csv',index = False)
    #result[series].to_csv('/Users/justinniestroy-admin/Desktop/UVA_' + str(id) + '_' + series + '_all.csv',index = False)
print('Finished Houlter Example')
############################
#
# Below is NICU Example
#
##############################
# filepath = '/Users/justinniestroy-admin/Desktop/NICU Vitals/UVA_6738_vitals.mat'
# df,time = read_in_NICU_file(filepath)
# time = time.to_numpy()
# num_cols = df.shape[1]
# time_series = {}
# for i in range(num_cols):
#     time_series[list(df.columns.values)[i]] = df[list(df.columns.values)[i]].to_numpy()
# interval_length = 60*10
# step_size = 60*5
# result = run.run_all(time_series,time)
# for series in result:
#     result[series].to_csv('/Users/justinniestroy-admin/Desktop/NICU Vitals/UVA_6738_' + series + '_all.csv',index = False)
# end_times = np.arange(np.min(time) + interval_length,np.max(time),step_size)
# for t in end_times:
#     test = run.get_interval(interval_length,end_times[1],time)
#     if len(test[0]) != 300:
#         print('Bad News')
