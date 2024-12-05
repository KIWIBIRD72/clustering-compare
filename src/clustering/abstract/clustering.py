from abc import ABC, abstractmethod
from scipy.sparse import spmatrix
from numpy.typing import NDArray
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


class Clustering(ABC):
    @abstractmethod
    def _dim_reduce(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Сжатие матрицы документов. Можно использовать разные методы сжатия.
        - Можно использовать разное сжатие из класса Reducer
        """
        pass

    @abstractmethod
    def _normalize_pixel(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Нормализация для каждого пикселя в матрице

        Returns:
            NDArray[np.float32]: _description_
        """
        pass

    # @abstractmethod
    # def plot_show(self, *args, **kwargs):
    #     """Отображение графического представления кластеров
    #     """
    #     pass

    @abstractmethod
    def cluster(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[RandomForestClassifier | KMeans, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        pass
