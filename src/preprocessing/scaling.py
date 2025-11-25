
from sklearn.preprocessing import StandardScaler, MinMaxScaler
def scale_standard(df):
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)
    return df, scaler
