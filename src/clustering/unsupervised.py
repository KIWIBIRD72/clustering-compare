import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from src.clustering.abstract.clustering import Clustering
from src.clustering.train_data import TrainData
from src.clustering.dim_reducer import DimReducer
from src.train_data_img.train_data_gen import TrainDataGen
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


class UnsupervisedClassifier(Clustering):
    TEST_SIZE = 0.2  # размер тестовой выборки
    RANDOM_STATE = 42
    CLUSTERS_AMOUNT = len(TrainDataGen.DIGITS)

    def __init__(self, n_clusters: int):
        self.__PIXEL_NOR_VAL = 255.0
        self.CLUSTERS_AMOUNT = n_clusters or self.CLUSTERS_AMOUNT
        self.__OPTIMAL_SVD_N = 10  # Оптимальное значение `n_components` для svd reduce

    def _normalize_pixel(self, X: NDArray[np.float64]):
        return X / self.__PIXEL_NOR_VAL

    def _dim_reduce(self, X:  NDArray[np.float64]):
        reducer = DimReducer()
        return reducer.tsne_reduce(X)

    def plot_clusters(self, X: NDArray[np.float64], y: NDArray[np.float64], title="Cluster Visualization"):
        reducer = DimReducer()
        # Уменьшаем размерность данных до 2D
        X_2d = reducer.tsne_reduce(X)

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                              c=y, cmap='viridis', s=150, alpha=0.7)
        plt.colorbar(scatter, label='Cluster Labels')

        # Добавление меток классов для каждой точки
        for i, label in enumerate(y):
            plt.text(
                X_2d[i, 0],
                X_2d[i, 1],
                f'{str(int(label))}',
                fontsize=6,
                ha='center',
                va='center',
                color='white',
                bbox=dict(facecolor='black', edgecolor='none',
                          alpha=0.6, boxstyle='round,pad=0.3'),
            )

        plt.title(title)
        plt.grid(True)
        plt.show()

    def cluster(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[KMeans, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        X_norm = self._normalize_pixel(X)
        X_reduced = self._dim_reduce(X_norm)

        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE)

        kmeans = KMeans(n_clusters=self.CLUSTERS_AMOUNT, random_state=42)
        kmeans.fit(X_train)

        return kmeans, X_test, y_test, y_train
