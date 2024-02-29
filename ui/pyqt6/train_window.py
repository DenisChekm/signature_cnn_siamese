import time
from datetime import datetime

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QWidget
from PyQt6.uic import loadUi

from model import siamese_bce
from ui.pyqt6.worker import my_worker, my_worker_with_params, worker_calculate_data


class TrainWindow(QWidget):
    def __init__(self):
        super(TrainWindow, self).__init__()
        loadUi("C:\\Users\\denle\\PycharmProjects\\signature_cnn_siamese\\ui\\pyqt6\\train_view.ui", self)

        # self.setWindowIcon(QIcon("../../images/signature-icon-50.png"))
        self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.push_button_train.clicked.connect(self.start_training_task)

    def __set_enabled_train_controls(self, isEnable: bool):
        self.push_button_train.setEnabled(isEnable)  # self.push_button_train.setText("Начать обучение")
        self.push_button_test_model.setEnabled(isEnable)
        self.group_box_train_params.setEnabled(isEnable)

    def __plot_graphs(self, train_losses, val_losses):
        train_avg_losses = train_losses["avg_loss"]  # {"train": {"std_loss": std_train_losses},
        val_avg_losses = val_losses["avg_loss"]
        pass

    def __show_report(self, report):
        acc = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1 - score']

    def __show_matrix(self, matrix):
        pass

    def __train_model(self):
        self.plain_text_edit_log.clear()
        res = siamese_bce.fit(self.spin_box_batch_size.value(), self.spin_box_epochs_count.value(),
                              self.plain_text_edit_log.appendPlainText)
        return res

    def __show_result(self, result):
        # train_data = result["train"]
        # val_data = result["val"]
        train_data, val_data, report, matrix = result

        self.__plot_graphs(train_data, val_data)

        # test_data = result["test"]
        # self.__show_report(test_data["report"])
        # self.__show_matrix(test_data["matrix"])

    def __thread_complete(self):
        self.plain_text_edit_log.appendPlainText(f"Конец обучения {datetime.now():%d.%m.%Y %H:%M:%S%z}")
        self.__set_enabled_train_controls(True)

    def start_training_task(self):
        self.__set_enabled_train_controls(False)

        worker = worker_calculate_data.Worker(self.__train_model)
        worker.signals.result.connect(self.__show_result)
        worker.signals.finished.connect(self.__thread_complete)

        # Execute
        self.threadpool.start(worker)
