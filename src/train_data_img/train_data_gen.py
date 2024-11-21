import os
from PIL import Image, ImageDraw, ImageFont
from constants.general import ROOT_DIR


class TrainDataGen:
    """Генерация изображений для тестирования кластеризации

    Генерация png изображений (чисел) 40x40 на черном фоне белыми буквами,
    разными шрифтами.
    """

    IMG_SIZE = (40, 40)
    DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self):
        self.__FONTS_DIR = ROOT_DIR / 'data' / 'fonts'
        self.__ALLOWED_FONT_EXTENSION = '.ttf'
        self.__OUTPUT_DIR = ROOT_DIR / 'data' / 'training' / 'numbers' / 'gen'
        # размер генерируемого изображения в пикселях

    def __get_fonts_paths(self):
        fonts: list[str] = []
        for font_name in os.listdir(self.__FONTS_DIR):
            if not font_name.endswith(self.__ALLOWED_FONT_EXTENSION):
                continue
            fonts.append(os.path.join(self.__FONTS_DIR, font_name))

        return fonts

    def __create_output_dir(self):
        if not os.path.exists(self.__OUTPUT_DIR):
            os.makedirs(self.__OUTPUT_DIR)

        return self.__OUTPUT_DIR

    def __gen_digit(self, amount_per_num: int, digit: int, fonts: list[str]):
        """Создание изображения для определенного числа

        Args:
            amount_per_num (int): кол-во изображений для числа
            digit (int): для какого числа генерировать изображение
            fonts (list[str]): массив путей для шрифтов
        """
        digit_dir = os.path.join(self.__OUTPUT_DIR, str(digit))
        os.makedirs(digit_dir, exist_ok=True)

        generated_img_path: list[str] = []

        for i in range(amount_per_num):
            # Создаем изображение с черным фоном
            img = Image.new('RGB', self.IMG_SIZE, "black")
            draw = ImageDraw.Draw(img)

            # Выбираем шрифт
            font_path = fonts[i]
            # Подбираем размер шрифта пропорционально размеру картинки
            font_size = int(self.IMG_SIZE[1] * 0.8)
            font = ImageFont.truetype(font_path, font_size)

            # Вычисляем размеры текста для центрирования
            text = str(digit)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (self.IMG_SIZE[0] - text_width) // 2
            y = (self.IMG_SIZE[1] - text_height * 1.5) // 2

            # Рисуем текст
            draw.text((x, y), text, font=font, fill="white")

            # Сохраняем изображение
            img_save_path = os.path.join(digit_dir, f"{digit}_{i + 1}.png")
            generated_img_path.append(img_save_path)
            img.save(img_save_path)

        return generated_img_path

    def get_fonts_amount(self):
        fonts = self.__get_fonts_paths()
        return len(fonts)

    def generate(self, amount_per_num: int) -> dict[str, list[str]]:
        """Генерация изображений для каждого числа от 0 до 9

        Args:
            amount_per_num (int): кол-во генерации изображений для каждого числа. 
                                  Не может быть больше, чем кол-во шрифтов в папке 
                                  со шрифтами `self.__FONTS_DIR`
        """
        self.__create_output_dir()
        fonts = self.__get_fonts_paths()

        if amount_per_num > len(fonts):
            raise Exception(f'[TrainDataImg > generate] amount_per_num: {
                            amount_per_num} can not be greater than fonts amount: {len(fonts)}')

        gen_img_dict = {}
        for digit in self.DIGITS:
            generated_digits = self.__gen_digit(
                amount_per_num, digit=digit, fonts=fonts)
            gen_img_dict[str(digit)] = generated_digits

        return gen_img_dict
