
import pandas as pd
from sklearn.impute import SimpleImputer

def fill_missing_numeric(df, strategy="mean"):
    imputer = SimpleImputer(strategy=strategy)
    num_df = df.select_dtypes(include='number')
    df[num_df.columns] = imputer.fit_transform(num_df)
    return df
