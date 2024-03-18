from datetime import datetime
import logging

from model import siamese_bce, siamese_dist, siamese_model, old_model_best_performance
from model.loss import euclidian_contrasive_loss, cosine_contrastive_loss
from utils.config import Config

PREDICT_FOLDER = "../sign_data/predict/"
BATCH_SIZE = 32
EPOCHS = 50


def configure_log():
    log_file_name = f"logs/checkmodel{datetime.now():%Y%m%d}.log"
    print(log_file_name)
    logging.basicConfig(filename=log_file_name, filemode='a', format='%(asctime)s, %(levelname)s %(message)s',
                        datefmt='%d.%m.%Y %H:%M:%S', level=logging.DEBUG, encoding="utf-16")


def predict_my_signature(model, output_fn):
    path_1 = PREDICT_FOLDER + "i am.jpg"
    path_2 = PREDICT_FOLDER + "f.jpg"

    res = model.predict(path_1, path_2)
    output_fn(res)

    path_2 = PREDICT_FOLDER + "m.jpg"
    res = model.predict(path_1, path_2)
    output_fn(res)

    res = model.predict(path_1, path_1)
    output_fn(res)


def train_test_predict_bce(output_fn_callback):
    siamese_bce.fit(BATCH_SIZE, EPOCHS, output_fn_callback)
    model = siamese_bce.SignatureNet.load_best_model()
    output_fn_callback(model)

    # model = siamese_bce.SignatureNet()
    # model.load_best_model()
    # predict_my_signature(model, output_fn)


def train_test_predict_dist(output_fn):
    output_fn(f"THRESHOLD =  {Config.THRESHOLD}")
    output_fn(f"margin = {euclidian_contrasive_loss.ContrastiveLoss().margin}")

    model = siamese_dist.SignatureNet()
    output_fn(model)
    model.fit(BATCH_SIZE, EPOCHS, output_fn)
    report, matrix = model.test(BATCH_SIZE, output_fn)
    output_fn(report)
    output_fn(matrix)

    # model = siamese_dist.SignatureNet()
    # model.load_best_model()
    # predict_my_signature(model, output_fn)


def train_test_predict_dist_model_old(output_fn):
    output_fn(f"> {old_model_best_performance.THRESHOLD}")
    output_fn(f"margin = {cosine_contrastive_loss.ContrastiveLoss().get_margin()}")
    model = old_model_best_performance.SiameseModel()
    output_fn(model)

    model.fit(BATCH_SIZE, EPOCHS, output_fn)
    report, matrix, report_a1, matrix_a1 = model.test(BATCH_SIZE, output_fn)

    output_fn(report)
    output_fn(matrix)
    output_fn("a1:")
    output_fn(report_a1)
    output_fn(matrix_a1)

    # model = old_model_best_performance.SignatureNet()
    # model.load_best_model()
    # predict_my_signature(model, output_fn)


def train():
    # SiameseModel.train_with_test()

    # train_test_predict_bce(logging.info)

    train_test_predict_dist(logging.info)

    # train_test_predict_dist_model_old(logging.info)


if __name__ == '__main__':
    configure_log()
    Config.seed_torch()

    train()
    # print(list(Config.divisor_generator(396))) # 486
