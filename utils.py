import numpy as np
import math
import pandas as pd


def append_or_add_key(d,key,val):
    try:
        d[key].append(val)
    except KeyError:
        d[key] = [val]

def build_innerxy_df(class_results_dict):
    dict_of_dicts = {}
    for k,v in class_results_dict.items():
        if isinstance(v,str):
            v = [float(x) for x in v[1:-1].split()] # Will be str if loaded from file
        ar = np.array([item for item in v if not math.isnan(item)])
        mean = ar.mean()
        var,std = (ar.var(), ar.std()) if len(v) > 1 else (0,0)
        dict_of_dicts[k] = {'mean':mean,'var':var,'std':std,'raw':ar}
    return dict_of_dicts

def results_dict_to_df(results_dict):
    """Converts and dict of dicts, with outer keys being methods and inner keys
    being classes, to a df with columns 'mean','var','std' and 'raw'
    """

    results_by_method = [pd.DataFrame(build_innerxy_df(d)).T for d in results_dict.values()]
    return pd.concat(results_by_method,axis=0,keys=results_dict.keys())

def filter_nans_from_df(df):
    """Takes a results df, which has a 'raw' column of all the individual
    results along with some stats such as mean. Removes all nans from the 'raw'
    column, recomputes the stats (which would all be nan if there were nans in
    'raw') and returns the result.
    """

    dict_of_dicts = {k:df['raw'].loc[k].to_dict() for k in df['raw'].index.levels[0]}
    return results_dict_to_df(dict_of_dicts)

def make_alls_df(df):
    inner_alls_dfs = [df.loc[k,'all'] for k in df.index.levels[0]]
    return pd.concat(inner_alls_dfs,keys=df.index.levels[0],axis=1).T
