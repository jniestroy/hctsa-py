import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
hr = pd.read_csv('/Users/justinniestroy-admin/Desktop/NICU Vitals/UVA_6738_RESP.csv')
hr = pd.read_csv('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/hctsa results/UVA0211_hist_test2.csv')
time = hr['time'].to_numpy() / 60 / 60
#print(np.max(time))
steps = (np.max(time) - np.min(time)) / 43
time2 = np.round(np.arange(np.min(time),np.max(time),steps))
#print(time2)
#hr2  = hr.drop(['Unnamed: 0','time','First moment'],axis = 1)
hr2  = hr.drop(['time','First moment','IsSeasonal?'],axis = 1)
#df_norm = (hr2 - hr2.mean()) / (hr2.std())
df_norm = (hr2 - hr2.mean()) / (hr2.max()-hr2.min())
#print(df_norm.shape)
plt.figure(figsize=(8, 12))

ax = sns.heatmap(df_norm,xticklabels=True)
plt.xlabel('Algorithm')
plt.ylabel('Hours since Midnight of Birthday')
ax.set_yticklabels(time2)
ax.tick_params(axis='both', which='major', labelsize=5)

plt.savefig('Houlter with mix of AF all.png',bbox_inches = "tight")
#plt.show()
