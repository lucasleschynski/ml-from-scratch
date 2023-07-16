from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from .utils import standardize

class PCA():
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.eigenvectors = None
        self.eigenvalues = None

    def fit(self, data: np.matrix[(Any, Any), np.float32]):
        if self.n_components > np.shape(data)[1]:
            raise ValueError("n_components is greater than size of feature vector")
        standardized_data = standardize(data)
        cov = np.cov(standardized_data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        return eigenvalues, eigenvectors
    
    def transform(self, data):
        standardized_data = standardize(data)
        top_k_vectors = (self.eigenvectors.T)[:, :self.n_components]
        transformed_data = np.matmul(standardized_data, top_k_vectors)
        return transformed_data
    
    def plot_pca_transform(self, data, labels):
        self.fit(data)
        transformed_data = self.transform(standardize(data))
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels)
        plt.show()
