import cv2
from PIL import Image

from PyQt6.QtCore import QThreadPool
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.uic import loadUi

from model import siamese_bce
from ui.pyqt6.worker import my_worker, my_worker_with_params, worker_calculate_data


class TrainWindow(QWidget):
    def __init__(self):
        super(TrainWindow, self).__init__()
        loadUi("C:\\Users\\denle\\PycharmProjects\\signature_cnn_siamese\\ui\\pyqt6\\predict_view.ui", self)

        # self.setWindowIcon(QIcon("../../images/signature-icon-50.png"))
        self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.push_button_train.clicked.connect(self.start_training_task)

    def __set_enabled_train_controls(self, isEnable: bool):
        self.push_button_train.setEnabled(isEnable)  # self.push_button_train.setText("Начать обучение")
        self.push_button_test_model.setEnabled(isEnable)
        self.group_box_train_params.setEnabled(isEnable)

    def load_image(self, file_name):
        pixmap = QPixmap(file_name)

        self.label = QLabel(self)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(), pixmap.height())

        self.resize(pixmap.width(), pixmap.height())

    def __transform_image(self, filename):
        image_bgr = cv2.imread(filename)
        new_size = (250, 250)
        image_bgr_resize = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_bgr_resize, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        return ImageTk.PhotoImage(image_pil)

    def load_image1(self):
        filename = filedialog.askopenfilename(title="Выбор изображения подписи",
                                              filetypes=[("Image files", ".png .jpg .jpeg"),
                                                         ("PNG", ".png"), ("JPG", ".jpg"), ("JPEG", ".jpeg")])
        if filename:
            self.filename1 = filename
            self.image1 = self.__transform_image(filename)
            self.canvas1.create_image(0, 0, anchor=NW, image=self.image1)

    def load_image2(self):
        filename = filedialog.askopenfilename(title="Выбор изображения подписи",
                                              filetypes=[("Image files", ".png .jpg .jpeg"),
                                                         ("PNG", ".png"), ("JPG", ".jpg"), ("JPEG", ".jpeg")])
        if filename:
            self.filename2 = filename
            self.image2 = self.__transform_image(filename)
            self.canvas2.create_image(0, 0, anchor=NW, image=self.image2)

    def make_prediction(self):
        if self.image1 is not None and self.image2 is not None:
            cos_sim, confidence, predicted_label = siamese_bce.predict(self.siamese_model,
                                                                        self.filename1,
                                                                        self.filename2)
            showinfo(title="Результат проверки", message=predicted_label)
            self.predict_result_label.configure(text=predicted_label)

        else:
            showerror(title="Ошибка", message="Нужно выбрать оба изображения")