
from sklearn.preprocessing import PolynomialFeatures
def add_polynomial_features(X, degree=2):
    return PolynomialFeatures(degree=degree).fit_transform(X)
