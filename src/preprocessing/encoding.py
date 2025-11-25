
import pandas as pd
from sklearn.preprocessing import LabelEncoder
def onehot_encode(df, cols):
    return pd.get_dummies(df, columns=cols)
