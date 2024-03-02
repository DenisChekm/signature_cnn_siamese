from PyQt6.QtCore import QThreadPool
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget, QLabel, QFileDialog, QMessageBox
from PyQt6.uic import loadUi

from model import siamese_bce
from ui.pyqt6.worker import worker_calculate_data

FILE_DIALOG_IMAGES_FILTER = "Images (*.png *.jpg *.jpeg);;PNG (*.png);JPG (*.jpg);;JPEG (*.jpeg)"
FILE_DIALOG_MODELS_FILTER = "Models *.pt"


class PredictWindow(QWidget):
    def __init__(self):
        super(PredictWindow, self).__init__()
        loadUi("C:\\Users\\denle\\PycharmProjects\\signature_cnn_siamese\\ui\\pyqt6\\predict_view.ui", self)

        # self.setWindowIcon(QIcon("../../images/signature-icon-50.png"))
        # self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.image_left_filename = None
        self.image_right_filename = None
        self.push_button_choose_image_left.clicked.connect(self.__load_left_image)
        self.push_button_choose_image_right.clicked.connect(self.__load_right_image)
        self.push_button_load_model.clicked.connect(self.__load_model)
        self.push_button_predict.clicked.connect(self.__make_prediction)

        self.model = siamese_bce.SignatureNet.load_best_model()

    # def __set_enabled_train_controls(self, isEnable: bool):
    #     self.push_button_train.setEnabled(isEnable)
    #     self.push_button_test_model.setEnabled(isEnable)
    #     self.group_box_train_params.setEnabled(isEnable)

    def __get_file(self, caption, files_filter):
        file_name, _ = QFileDialog.getOpenFileName(self, caption=caption, directory="C:/", filter=files_filter)
        return file_name

    def load_image(self, file_name, label: QLabel):
        self.line_edit_prediction_result.clear()
        label.setPixmap(QPixmap(file_name))

    def __load_left_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выбор изображения подписи", "C:/", FILE_DIALOG_IMAGES_FILTER)
        if file_name:
            self.image_left_filename = file_name
            self.load_image(file_name, self.label_left_image)

    def __load_right_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выбор изображения подписи", "C:/", FILE_DIALOG_IMAGES_FILTER)
        if file_name:
            self.image_right_filename = file_name
            self.load_image(file_name, self.label_right_image)

    def __load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выбор файла обученной модели сети", "C:/",
                                                   FILE_DIALOG_MODELS_FILTER)
        if file_name:
            self.model = siamese_bce.SignatureNet.load_model(file_name)

    def __make_prediction(self):
        if self.image_left_filename is not None and self.image_right_filename is not None:
            predicted_label = siamese_bce.predict(self.model, self.image_left_filename, self.image_right_filename)
            self.line_edit_prediction_result.setText(predicted_label)
            QMessageBox.information(self, "Результат", predicted_label)
        else:
            QMessageBox.warning(self, "Предупреждение", "Нужно выбрать оба изображения")
