
from sklearn.linear_model import LinearRegression
import numpy as np

def test_linear_regression_train():
    X = np.array([[1],[2],[3],[4]])
    y = np.array([2,4,6,8])
    model = LinearRegression().fit(X,y)
    preds = model.predict(np.array([[5]]))
    assert abs(preds[0]-10) < 1e-6
