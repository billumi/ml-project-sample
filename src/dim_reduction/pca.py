
from sklearn.decomposition import PCA
def apply_pca(X,n): p=PCA(n_components=n); return p.fit_transform(X), p
