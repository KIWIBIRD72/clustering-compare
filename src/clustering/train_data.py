import os
import numpy as np
import pandas as pd
from constants.general import ROOT_DIR
from numpy.typing import NDArray
from PIL import Image
from src.train_data_img.train_data_gen import TrainDataGen


class TrainData:
    def __init__(self):
        self.__HANDWRITTEN_TRAIN_DATA = ROOT_DIR / 'data' / 'training' / 'numbers'
        self.__CANDLES_DATA_PATH = ROOT_DIR / 'data' / \
            'training' / 'candles' / 'data.csv'
        self.__ALLOWED_EXTENSIONS = ('.jpg', '.png')

    def __transform_data(self, train_data_paths: dict[str, list[str]]):
        """Загрузка изображений и серялизация данных в массив пикселей и labels

        :return: Кортеж (X, y), где X — массив пикселей, y — метки классов.
        """
        X: list[NDArray[np.float64]] = []
        y: list[NDArray[np.float64]] = []

        for label, images in train_data_paths.items():
            for file_path in images:
                try:
                    img = Image.open(file_path).convert(
                        'L').resize(TrainDataGen.IMG_SIZE)
                    # Добавляем векторное изображение и label для класса
                    X.append(np.array(img, dtype=np.float64).flatten())
                    y.append(np.array(int(label), dtype=np.float64))
                except Exception as error:
                    print(f'[_vectorize] Error while opening image at path: {
                          file_path}')
        return np.array(X), np.array(y)

    def __get_images_data(self):
        """Объединение сгенерированных данных для обучения и рукописных данных

        Returns:
            Dict, где ключ - label для класса, а значение - массив с path изображений
        """
        tran_data_generator = TrainDataGen()
        fonts_amount = tran_data_generator.get_fonts_amount()
        gen_imgs = tran_data_generator.generate(fonts_amount)

        # add handwritten classes
        for label in os.listdir(self.__HANDWRITTEN_TRAIN_DATA):
            label_dir = os.path.join(self.__HANDWRITTEN_TRAIN_DATA, label)
            if not os.path.isdir(label_dir):
                continue

            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                if not file_path.lower().endswith(self.__ALLOWED_EXTENSIONS):
                    continue

                gen_imgs[label] = gen_imgs[label] + [file_path]
        return gen_imgs

    def get_imgs(self):
        train_data = self.__get_images_data()
        X, y = self.__transform_data(train_data)

        return X, y

    def get_candels(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Загрузка данных из CSV-файла со свечами и преобразование их в тренировочный формат.
        Target - метки классов: 1, если цена выросла; 0, если упала

        :return: Кортеж (X, y), где X — массив признаков, y — метки классов.
        """
        try:
            file_path = self.__CANDLES_DATA_PATH
            df = pd.read_csv(file_path)

            features = ['open', 'high', 'low', 'close', 'volume', 'quoteVolume',
                        'trades', 'baseAssetVolume', 'quoteAssetVolume', 'dayOfWeek', 'month']
            target = 'close'

            for column in features + [target]:
                if column not in df.columns:
                    raise ValueError(
                        f"Column '{column}' not found in the CSV file.")

            day_of_week_mapping = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            df['dayOfWeek'] = df['dayOfWeek'].map(day_of_week_mapping)

            df['openTime'] = pd.to_datetime(df['openTime'], unit='ms')
            df['month'] = df['openTime'].dt.month

            df = df.dropna(subset=features + [target])

            # Генерация меток классов: 1, если цена выросла; 0, если упала
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

            # Удаляем последнюю строку, так как для неё нет метки
            df = df[:-1]
            print(df)

            X = df[features].to_numpy(dtype=np.float64)
            y = df['target'].to_numpy(dtype=np.float64)

            return X, y

        except Exception as e:
            print(f"[get_candels] Error: {e}")
            return np.array([]), np.array([])
