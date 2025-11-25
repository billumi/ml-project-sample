
import pandas as pd
def equal_width_binning(df, col, bins):
    df[col+'_bin']=pd.cut(df[col], bins=bins)
    return df
