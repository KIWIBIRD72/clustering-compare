from src.train_data_img.train_data_gen import TrainDataGen


def main():
    train_data_gen = TrainDataGen()

    fonts_amount = train_data_gen.get_fonts_amount()
    gen_imgs = train_data_gen.generate(fonts_amount)
    print(gen_imgs)


if __name__ == '__main__':
    main()
