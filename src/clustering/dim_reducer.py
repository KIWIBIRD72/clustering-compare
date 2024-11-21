import numpy as np
from scipy.sparse import spmatrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from numpy.typing import NDArray


class DimReducer:
    """Алгоритмы уменьшения размеров матриц

    See Also
    --------
        - https://github.com/KIWIBIRD72/clustering/blob/main/assets/simplified%20clustering.ipynb
    """

    def pca_reduce(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        reducer = PCA(n_components=2)
        return reducer.fit_transform(X)

    def tsne_reduce(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        reducer = TSNE(
            init="random",
            n_components=2,          # 2D for visualization
            perplexity=14,           # Lower value to focus on local structure
            learning_rate=100,       # Higher learning rate for cluster exaggeration
            early_exaggeration=10,   # Increase early exaggeration
            n_iter=500,              # More iterations for better convergence
            random_state=42,
        )
        return reducer.fit_transform(X)

    def svd_reduce(self, X: NDArray[np.float64], n_components: int) -> NDArray[np.float64]:
        reducer = TruncatedSVD(n_components=n_components)
        return reducer.fit_transform(X)
