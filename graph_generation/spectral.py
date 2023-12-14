import warnings

import numpy as np
from scipy.sparse import coo_matrix, csgraph, linalg


class SpectrumExtractor:
    def __init__(self, num_features, normalized=True):
        self.num_features = num_features
        self.normalized = normalized

    def __call__(self, adj):
        n = adj.shape[0]
        adj = coo_matrix(adj, copy=True, dtype=np.float64)
        lap = csgraph.laplacian(adj, normed=self.normalized, copy=False)

        # get first k non-zero eigenvalues and corresponding eigenvectors sorted in ascending order of eigenvalues
        # in case k is larger than the number of nodes, fill up with zeros
        eigenvalues_full = np.zeros(self.num_features)
        eigenvectors_full = np.zeros((n, self.num_features))
        k = min(n - 2, self.num_features)
        if k > 0:
            while True:
                try:
                    eigenvalues, eigenvectors = linalg.eigsh(lap, k=k + 1, which="SM")
                    break
                except:
                    warnings.warn("eigsh failed to converge, trying again")

            eigenvalues_full[:k] = eigenvalues[1:]
            eigenvectors_full[:, :k] = eigenvectors[:, 1:]
        eigenvalues_repeated = eigenvalues_full[np.newaxis, :].repeat(n, axis=0)
        return np.concatenate((eigenvalues_repeated, eigenvectors_full), axis=1)
