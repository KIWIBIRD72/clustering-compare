import numpy as np
import matplotlib.pyplot as plt
from clustering.abstract.clustering import Clustering
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.clustering.dim_reducer import DimReducer
from src.clustering.train_data import TrainData


class SupervisedClassifier(Clustering):
    TEST_SIZE = 0.2  # размер тестовой выборки
    RANDOM_STATE = 42

    def __init__(self):
        self.__PIXEL_NORM_VAL = 255.0
        self.__OPTIMAL_SVD_N = 10  # Оптимальное значение `n_components` для svd reduce

    def _normalize_pixel(self, X: NDArray[np.float64]):
        return X / self.__PIXEL_NORM_VAL

    def _dim_reduce(self, X:  NDArray[np.float64]):
        reducer = DimReducer()
        return reducer.tsne_reduce(X)

    def plot_clusters(self, X: NDArray[np.float64], y: NDArray[np.float64], title="Cluster Visualization"):
        reducer = DimReducer()
        # Уменьшение размерности данных до 2D
        X_2d = reducer.svd_reduce(X, n_components=self.__OPTIMAL_SVD_N)

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                              c=y, cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(scatter, label='Cluster Labels')

        # Добавление меток классов для каждой точки
        int_y = y.astype(np.int64)
        for i, label in enumerate(int_y):
            plt.text(
                X_2d[i, 0],
                X_2d[i, 1],
                str(label),
                fontsize=7,
                ha='center',
                va='center',
                color='white',
            )

        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True)
        plt.show()

    def cluster(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[RandomForestClassifier, NDArray[np.float64], NDArray[np.float64], None]:
        X_norm = self._normalize_pixel(X)
        X_reduced = self._dim_reduce(X_norm)

        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model, X_test, y_test, None
