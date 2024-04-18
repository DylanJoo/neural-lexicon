# Try other kmean algors
# Try minibatch kmeans if needed
import faiss

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, device='cpu'):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None

        self.gpu = False if device == 'cpu' else True

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        # self.kmeans.train(X.astype(np.float32))
        self.kmeans.train(X)
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def assign(self, X):
        return self.kmeans.index.search(X, 1)[1].flatten()
