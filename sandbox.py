from datetime import datetime
import logging

from model import siamese_bce, siamese_dist

PREDICT_FOLDER = "../sign_data/predict/"
BATCH_SIZE = 32
EPOCHS = 50


def train_test_predict_bce(output_fn_callback):
    siamese_bce.fit(BATCH_SIZE, EPOCHS, output_fn_callback)
    model = siamese_bce.SignatureNet.load_best_model()
    output_fn_callback(model)

    # path_1 = PREDICT_FOLDER + "i am.jpg"
    # path_2 = PREDICT_FOLDER + "f.jpg"
    # output_fn_callback(siamese_bce.predict(model, path_1, path_2))
    #
    # path_2 = PREDICT_FOLDER + "m.jpg"
    # output_fn_callback(siamese_bce.predict(model, path_1, path_2))


def train_test_predict_dist(output_fn):
    output_fn(f"> {siamese_dist.THRESHOLD}")
    model = siamese_dist.SignatureNet()
    output_fn(model)

    model.fit(BATCH_SIZE, EPOCHS, output_fn)
    report, matrix = model.test(BATCH_SIZE, output_fn)
    output_fn(report)
    output_fn(matrix)

    # path_1 = PREDICT_FOLDER + "i am.jpg"
    # path_2 = PREDICT_FOLDER + "f.jpg"
    # model = siamese_dist.SignatureNet.load_best_model()
    # output_fn_callback(model.predict(path_1, path_2))
    #
    # path_2 = PREDICT_FOLDER + "m.jpg"
    # output_fn_callback(model.predict(path_1, path_2))


def train():
    # SiameseModel.train_with_test()

    log_file_name = f"logs/checkmodel{datetime.now():%Y%m%d}.log"
    print(log_file_name)
    logging.basicConfig(filename=log_file_name, filemode='a', format='%(asctime)s, %(levelname)s %(message)s',
                        datefmt='%d.%m.%Y %H:%M:%S', level=logging.DEBUG, encoding="utf-16")

    # train_test_predict_bce(logging.info)
    train_test_predict_dist(logging.info)


    # _ = SiameseModel.test_model_best_loss()


if __name__ == '__main__':
    train()
