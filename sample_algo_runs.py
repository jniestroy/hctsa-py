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

def read_in_NICU_file(path):
    arrays = {}
    f = h5py.File(path,'r')
    for k, v in f.items():
        if k != 'vdata' and k != 'vt':
            continue
        arrays[k] = np.array(v)
    df = pd.DataFrame(np.transpose(arrays['vdata']))
    df = df.dropna(axis=1, how='all')
    df.columns = get_column_names(f['vname'],f)
    times = pd.Series(arrays['vt'][0], index=df.index)
    df

    return df,times
def get_column_names(vname,f):
    names = []
    for name in vname[0]:
        obj = f[name]
        col_name = ''.join(chr(i) for i in obj[:])
        names.append(col_name)
    return names


pid = '1282'
filepath = '/Users/justinniestroy-admin/Desktop/PreVent/UVA_'+pid + '_vitals.mat'
#filepath = '/Users/justinniestroy-admin/Desktop/NICU Vitals/UVA_6738_vitals.mat'
df,time = read_in_NICU_file(filepath)
time = time.to_numpy()
num_cols = df.shape[1]
time_series = {}
for i in range(num_cols):
    if list(df.columns.values)[i] == 'HR' or list(df.columns.values)[i] == 'RESP' or list(df.columns.values)[i] == 'SPO2-%':
        time_series[list(df.columns.values)[i]] = df[list(df.columns.values)[i]].to_numpy()
interval_length = 60*10
step_size = 60*5
result = run.run_all(time_series,time,int(pid),algos ='all')
# for series in result:
#     result[series].to_csv('/Users/justinniestroy-admin/Desktop/NICU Vitals/UVA_6738_' + series + '_all.csv',index = False)
    #to add post meta data to mongo for each output
#Reads in new nicu data from Doug
# def read_in_NICU_file(path):
#     arrays = {}
#     f = h5py.File(path,'r')
#     for k, v in f.items():
#         arrays[k] = np.array(v)
#
#     df = pd.DataFrame(np.transpose(arrays['vdata']),columns = get_column_names(f['vname'],f))
#     times = pd.Series(arrays['vt'][0], index=df.index)
#
#     return df,times
#
# def get_column_names(vname,f):
#     names = []
#     for name in vname[0]:
#         obj = f[name]
#         col_name = ''.join(chr(i) for i in obj[:])
#         names.append(col_name)
#     return names

########################
#
#Houlter Data way to run
#
########################
# id = 7
# time, hr = read_in_data(id)
#
# time_series = {'hr':hr}
# interval_length = 60*10
# step_size = 60*5
# result = run.run_all(time_series,time)
# for series in result:
#     test = result[series].head()
#     test.to_csv('/Users/justinniestroy-admin/Desktop/UVA_' + str(id) + '_' + series + '_all.csv',index = False)
#     #result[series].to_csv('/Users/justinniestroy-admin/Desktop/UVA_' + str(id) + '_' + series + '_all.csv',index = False)
# print('Finished Houlter Example')
############################
#
# Below is NICU Example
#
##############################
# minioClient = Minio('minionas.uvadcos.io',
#                     access_key='breakfast',
#                     secret_key='breakfast',
#                     secure=False)
# objects = minioClient.list_objects('breakfast',
#                               recursive=True)
# for obj in objects:
#     #print(obj.bucket_name, obj.object_name.encode('utf-8'), obj.last_modified,
#      #     obj.etag, obj.size, obj.content_type)
#     print(obj.object_name)
#
# spark = SparkSession.builder
#                         .master("local")
#                         .appName("app name")
#                         .config("spark.some.config.option", true).getOrCreate()
#
# df = spark.read.parquet("s3://path/to/parquet/file.parquet")
