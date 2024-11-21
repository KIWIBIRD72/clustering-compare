import os
import numpy as np
from constants.general import ROOT_DIR
from numpy.typing import NDArray
from PIL import Image
from src.train_data_img.train_data_gen import TrainDataGen


class TrainData:
    def __init__(self):
        self.__HANDWRITTEN_TRAIN_DATA = ROOT_DIR / 'data' / 'training' / 'numbers'
        self.__ALLOWED_EXTENSIONS = ('.jpg', '.png')

    def __transform_data(self, train_data_paths: dict[str, list[str]]):
        """Загрузка изображений и серялизация данных в массив пикселей и labels

        :return: Кортеж (X, y), где X — массив пикселей, y — метки классов.
        """
        X: list[NDArray[np.float64]] = []
        y: list[NDArray[np.int8]] = []

        for label, images in train_data_paths.items():
            for file_path in images:
                try:
                    img = Image.open(file_path).convert(
                        'L').resize(TrainDataGen.IMG_SIZE)
                    # Добавляем векторное изображение и label для класса
                    X.append(np.array(img, dtype=np.float64).flatten())
                    y.append(np.array(int(label), dtype=np.int8))
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

    def get(self):
        train_data = self.__get_images_data()
        X, y = self.__transform_data(train_data)

        return X, y
