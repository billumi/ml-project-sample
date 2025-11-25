
import pandas as pd
from src.preprocessing.missing_values import fill_missing_numeric

def test_fill_missing_numeric():
    df = pd.DataFrame({'a':[1, None, 3], 'b':[4,5, None]})
    out = fill_missing_numeric(df.copy(), strategy='mean')
    assert not out.isnull().any().any()
