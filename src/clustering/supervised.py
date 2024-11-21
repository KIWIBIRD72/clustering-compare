import numpy as np
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
        return reducer.svd_reduce(X, n_components=self.__OPTIMAL_SVD_N)

    def cluster(self) -> tuple[RandomForestClassifier, NDArray[np.float64], NDArray[np.float64], None]:
        train_data = TrainData()
        X, y = train_data.get()

        X_norm = self._normalize_pixel(X)
        X_reduced = self._dim_reduce(X_norm)

        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model, X_test, y_test, None
