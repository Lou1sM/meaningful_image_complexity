import pandas as pd
import numpy as np
import math
import sys
from utils import filter_nans_from_df,make_alls_df
from project_config import DSETS

#dsets = ['im','cifar','dtd','mnist','stripes','halves','rand']

exp_name = sys.argv[1]

serieses = []
for d in DSETS:
    print(d)
    df = pd.read_csv(f'experiments/{exp_name}/{d}/{d}_results_by_class.csv',index_col=[0,1])
    df = filter_nans_from_df(df)
    df_alls = make_alls_df(df)
    #df_alls = df_alls.drop('raw',axis=1)
    try:
        df_no_mdl = pd.read_csv(f'experiments/{exp_name}/mdl_abls/{d}_no_mdl_abl.csv',index_col=[0])
        df_no_mdl.index = ['no_mdl']
        df_alls = pd.concat([df_alls,df_no_mdl],axis=0)
    except:
        print('cannot do',d)
    multi_index = pd.MultiIndex.from_product([df_alls.index,df_alls.columns],
                                                names=['method','metric'])
    new_series = pd.Series(df_alls.values.flatten(),index=multi_index)
    serieses.append(new_series)
df = pd.concat(serieses,keys=DSETS,axis=1)
breakpoint()

maxes=[df.loc[k].loc['mean'].max() for k in df.index.levels[0]]
normed=[df.loc[k]/df_max for k,df_max in zip(df.index.levels[0],maxes)]

df = pd.concat(normed,axis=0,keys=df.index.levels[0])

df = pd.concat([df.loc[k].T for k in df.index.levels[0]],keys=df.index.levels[0],axis=0)
df = df.astype(float).round(2)

# Insert repeated experiments
#df = df.drop('gclm')
df = df.rename(index={'gclm':'glcm'})
main_methods = ['patch_ent','ent','fract','glcm','jpg','khan','mach','redies']
abl_methods = ['patch_ent','no_mdl','ncs','mdl']
scale_methods = [f'patch_ent{i}' for i in range(4)]
print(df)
df.to_csv(f'experiments/{exp_name}/all_results.csv')

def both_mean_std_as_array(df):
    return np.array([f'{m} ({v})' for m,v in zip(df['mean'],df['std'])])

df=df.drop(['var'],axis=1)
df_main=df.loc[main_methods]
both_as_arr = both_mean_std_as_array(df_main).reshape(len(main_methods),7)
df_main=pd.DataFrame(both_as_arr,index=main_methods,columns=DSETS)

glcm_proper = pd.read_csv('experiments/main_run/glcms/all_dsets.csv',index_col=0)
glcm_proper = glcm_proper[glcm_proper.columns[0]]
df_main.loc['glcm'] = glcm_proper

#df_main = df_main.drop('redies')
redies_proper = pd.read_csv('experiments/main_run/redies/all_dsets.csv',index_col=0)
redies_proper = redies_proper[redies_proper.columns[0]]
df_main.loc['redies'] = redies_proper

#df_main = df_main.drop('mach')
mach_proper = pd.read_csv('experiments/main_run/mach/all_dsets.csv',index_col=0)
mach_proper = mach_proper[mach_proper.columns[0]]
df_main.loc['mach'] = mach_proper

df_abls=df.loc[abl_methods]
both_as_arr = both_mean_std_as_array(df_abls).reshape(len(abl_methods),7)
df_abls=pd.DataFrame(both_as_arr,index=abl_methods,columns=DSETS)

df_scales=df.loc[scale_methods]
both_as_arr = both_mean_std_as_array(df_scales).reshape(len(scale_methods),7)
df_scales=pd.DataFrame(both_as_arr,index=scale_methods,columns=DSETS)

df_main.to_csv(f'experiments/{exp_name}/main_results.csv')
df_abls.to_csv(f'experiments/{exp_name}/ablation_results.csv')
df_scales.to_csv(f'experiments/{exp_name}/scales_results.csv')
