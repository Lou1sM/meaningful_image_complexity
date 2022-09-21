import pandas as pd

dsets = ['im','cifar','dtd','mnist','stripes','halves','rand']
df = pd.concat([pd.read_csv(f'experiments/main_run4/{d}/{d}_results.csv',index_col=0).loc['means'] for d in dsets],keys=dsets,axis=1)
df = df.T.drop(['proc_time','img_label']+[f'level {i}' for i in range(1,5)],axis=1)

maxes = pd.concat([pd.read_csv(f'experiments/main_run4/{d}/{d}_results.csv',index_col=0).drop(['means','stds'],axis=0) for d in dsets],keys=dsets,axis=0).drop(['proc_time','img_label']+[f'level {i}' for i in range(1,5)],axis=1).max(axis=0)

df /= maxes

df.T.style.to_latex('full_main_results_latex.txt',float_format=lambda x: f'{x:.2f}',bold_rows=True)
