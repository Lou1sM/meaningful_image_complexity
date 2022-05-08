import pandas as pd

dsets = ['im','cifar','dtd','mnist','stripes','halves','rand']

exp_name = 'jim'

serieses = []
for d in dsets:
    df = pd.read_csv(f'experiments/{exp_name}/{d}_results.csv',index_col=[0])
    multi_index = pd.MultiIndex.from_product([df.index,df.columns],names=['method','metric'])
    new_series = pd.Series(df.values.flatten(),index=multi_index)
    serieses.append(new_series)
df = pd.concat(serieses,keys=dsets,axis=1)
x = df.values - df.values.min(axis=1,keepdims=True)
x /= x.max(axis=1,keepdims=True)+1e-8
df = pd.DataFrame(x,index=df.index,columns=df.columns)
df = pd.concat([df.loc[k].T for k in df.index.levels[0]],keys=df.index.levels[0],axis=0)
df = df.round(3)
print(df)

df.to_csv(f'experiments/{exp_name}/all_dsets.csv')
