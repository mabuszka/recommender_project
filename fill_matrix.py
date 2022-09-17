import numpy as np
import pandas as pd
import scipy.stats as sstats


# fill with value "a"
def fill_const(df, const):
    df_filled = df.fillna(const)
    return df_filled

# fill with mean/median/mode of all

def fill_all_stat(df, stat):
    if stat == "mean":
        df_filled = df.fillna(np.nanmean(df))
    elif stat == "mode":
        df_filled = df.fillna(sstats.mode(df, axis = None, nan_policy="omit")[0][0]) # we choose the smallest in case of multiple modes
    elif stat == "median":
        df_filled = df.fillna(np.nanmedian(df))
    else:
        raise Exception('Unrecognized stat')
    return(df_filled)

def fill_col_stat(df, stat, failsave=True):
    if stat == "mean":
        df_filled = df.fillna(df.mean(axis = 0))
    elif stat == "mode":
        df_filled = df.fillna(df.mode(axis = 0).iloc[0])  # we choose the smallest in case of multiple modes
    elif stat == "median":
        df_filled = df.fillna(df.median(axis = 0))
    else:
        raise Exception('Unrecognized stat')
    if failsave:
        df_filled = fill_all_stat(df_filled, stat) # failsave for cases where a column is all nan
    return df_filled

def fill_row_stat(df, stat, failsave=True):
    if stat == "mean":
        df_filled= df.mask(df.isna(), df.mean(axis=1), axis=0) 
    elif stat == "mode":
        df_filled = df.mask(df.isna(), df.mode(axis=1).iloc[:,0], axis=0)  
    elif stat == "median":
        df_filled = df.mask(df.isna(), df.median(axis=1), axis=0)   
    else:
        raise Exception('Unrecognized stat')
    if failsave:
       df_filled =  fill_all_stat(df_filled, stat) # failsave for cases where a row is all nan
    return df_filled

def fill_all_mean(df):
    df_filled = fill_all_stat(df, 'mean')
    return df_filled

def fill_from_distr(df, axis, failsave_val = None, random_state = None):
    n_row , n_col = df.shape
    if failsave_val is None:
        failsave_val = np.nanmean(df)
    if axis == "cols":
        val_dict = {}
        for col in df.columns:
            col_distr = df[col].dropna()
            if col_distr.size < 1:
                col_distr = pd.Series([failsave_val])
            col_sample = col_distr.sample(n = n_row, replace = True, random_state=random_state)
            val_dict[col] =  col_sample.set_axis(df[col].index)
        df_filled = df.fillna(pd.DataFrame(val_dict))
        # distr_dict = df.to_dict("list")
        # for key in distr_dict.keys():
        #     distr_dict[key] = np.array(distr_dict[key])[~np.isnan(distr_dict[key])]

    elif axis == "rows":
        val_np = df.to_numpy(copy = True)
        for row in range(n_row):
            row_vals = val_np[row,][~np.isnan(val_np[row,])]
            if row_vals.size < 1:
                row_vals = np.array([failsave_val])
            row_vals = np.random.choice(row_vals, n_col, replace = True)
            val_np[row,] = row_vals
        # print(val_np)
        # print(pd.DataFrame(val_np, index = df.index, columns=df.columns))
        # print(df)
        df_filled = df.fillna(pd.DataFrame(val_np, index = df.index, columns=df.columns))
    # else:
    #     print("not implemented")
    return df_filled
    
def fill_weighted_mean(df, row_weight):
    mean = np.nanmean(df)
    col_mean = df.mean(axis = 0).to_numpy()[np.newaxis]
    col_mean[np.isnan(col_mean)] = mean
    row_mean = df.mean(axis = 1).to_numpy()[np.newaxis].T
    row_mean[np.isnan(row_mean)] = mean
    fill = (1-row_weight) * col_mean + row_weight * row_mean
    df_filled = df.fillna(pd.DataFrame(fill, index = df.index, columns=df.columns))
    return df_filled
    


def fill_all_mean(df):
    df_filled = fill_all_stat(df, 'mean')
    return df_filled