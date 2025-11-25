
from sklearn.cluster import KMeans
def train_kmeans(X,k): m=KMeans(n_clusters=k); m.fit(X); return m
