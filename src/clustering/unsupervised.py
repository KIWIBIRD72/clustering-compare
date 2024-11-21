import numpy as np
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

    def __init__(self):
        self.__PIXEL_NOR_VAL = 255.0

    def _normalize_pixel(self, X: NDArray[np.float64]):
        return X / self.__PIXEL_NOR_VAL

    def _dim_reduce(self, X:  NDArray[np.float64]):
        reducer = DimReducer()
        return reducer.tsne_reduce(X)

    def cluster(self) -> tuple[KMeans, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        train_data = TrainData()
        X, y = train_data.get()

        X_norm = self._normalize_pixel(X)
        X_reduced = self._dim_reduce(X_norm)

        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE)

        kmeans = KMeans(n_clusters=self.CLUSTERS_AMOUNT, random_state=42)
        kmeans.fit(X_train)

        return kmeans, X_test, y_test, y_train
