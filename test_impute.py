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
    return df,times
def get_column_names(vname,f):
    names = []
    for name in vname[0]:
        obj = f[name]
        col_name = ''.join(chr(i) for i in obj[:])
        names.append(col_name)
    return names
filepath = '/Users/justinniestroy-admin/Desktop/PreVent/UVA_1282_vitals.mat'
#filepath = '/Users/justinniestroy-admin/Desktop/NICU Vitals/UVA_6738_vitals.mat'
df,time = read_in_NICU_file(filepath)
t = 224214
indx = run.get_interval(60*10,int(t),time)
indx = indx[0]
y = df['HR'].values[indx]
y_imputed = run.impute(y,155)
print(y_imputed)
