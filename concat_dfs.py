import pandas as pd

dsets = ['im','rand','cifar','mnist','simp']

dfs = [pd.read_csv(f'experiments/{d}_results.csv', index_col=[0,1]) for d in dsets]
concatted_df = pd.concat(dfs,axis=1,keys=dsets)
breakpoint()
concatted_df.to_csv('experiments/all_dsets_results.csv')
