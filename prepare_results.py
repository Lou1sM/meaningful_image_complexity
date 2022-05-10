import pandas as pd
import math
import sys
from utils import filter_nans_from_df,make_alls_df

dsets = ['im','cifar','dtd','mnist','stripes','halves','rand']

exp_name = sys.argv[1]

serieses = []
for d in dsets:
    print(d)
    df = pd.read_csv(f'experiments/{exp_name}/{d}/{d}_results_by_class.csv',index_col=[0,1])
    df = filter_nans_from_df(df)
    df_alls = make_alls_df(df)
    df_alls = df_alls.drop('raw',axis=1)
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
df = pd.concat(serieses,keys=dsets,axis=1)

maxes=[df.loc[k].loc['mean'].max() for k in df.index.levels[0]]
normed=[df.loc[k]/df_max for k,df_max in zip(df.index.levels[0],maxes)]

df = pd.concat(normed,axis=0,keys=df.index.levels[0])

df = pd.concat([df.loc[k].T for k in df.index.levels[0]],keys=df.index.levels[0],axis=0)
df = df.astype(float).round(4)
print(df)

df.to_csv(f'experiments/{exp_name}/all_dsets.csv')
