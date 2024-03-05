from datetime import datetime

from PyQt6.QtCore import QThreadPool
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget, QTableWidget, QTableWidgetItem
from PyQt6.uic import loadUi
from pyqtgraph import PlotWidget, mkPen
from sklearn.metrics import ConfusionMatrixDisplay

from model import siamese_dist
from ui.pyqt6.worker import worker_calculate_data

BLUE_PEN = mkPen(color=(66, 133, 180), width=4)
ORANGE_PEN = mkPen(color=(255, 165, 0), width=4)


def init_plot(plot_widget: PlotWidget, plot_title: str, y_axis_name: str):
    plot_widget.setBackground("w")
    plot_widget.setTitle(plot_title)
    plot_widget.setLabel("bottom", "эпоха")
    plot_widget.setLabel("left", y_axis_name)
    plot_widget.addLegend()
    plot_widget.showGrid(x=True, y=True)


def plot_graph(plot_widget: PlotWidget, train_values, val_values):
    plot_widget.plot(train_values, name="Обучение", pen=BLUE_PEN, symbol="+")
    plot_widget.plot(val_values, name="Валидация", pen=ORANGE_PEN, symbol="+")


class TrainWindow(QWidget):
    def __init__(self):
        super(TrainWindow, self).__init__()
        loadUi("./ui/pyqt6/train_view.ui", self)

        # self.setWindowIcon(QIcon("../../images/signature-icon-50.png"))
        self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.push_button_train.clicked.connect(self.start_training_task)

        init_plot(self.widget_plot_loss, "График ошибки", "Средняя ошибка")
        init_plot(self.widget_plot_acc, "График аккуратности", "Точность")

    def __set_enabled_train_controls(self, isEnable: bool):
        self.push_button_train.setEnabled(isEnable)  # self.push_button_train.setText("Начать обучение")
        self.push_button_test_model.setEnabled(isEnable)
        self.group_box_train_params.setEnabled(isEnable)

    def __plot_graphs(self, avg_train_losses, avg_val_losses, std_train_losses, std_val_losses):
        plot_graph(self.widget_plot_loss, avg_train_losses, avg_val_losses)
        plot_graph(self.widget_plot_acc, std_train_losses, std_val_losses)

    def __set_table_metrics(self, report: dict):
        label_real = report['0.0']
        label_forged = report['1.0']
        acc = report['accuracy']
        self.table_widget_report.setItem(0, 2, QTableWidgetItem(f"{label_real['precision']:.4f}"))
        self.table_widget_report.setItem(0, 3, QTableWidgetItem(f"{label_real['recall']:.4f}"))
        self.table_widget_report.setItem(0, 4, QTableWidgetItem(f"{label_real['f1-score']:.4f}"))

        self.table_widget_report.setItem(1, 2, QTableWidgetItem(f"{label_forged['precision']:.4f}"))
        self.table_widget_report.setItem(1, 3, QTableWidgetItem(f"{label_forged['recall']:.4f}"))
        self.table_widget_report.setItem(1, 4, QTableWidgetItem(f"{label_forged['f1-score']:.4f}"))

    def __show_confusion_matrix(self, matrix):
        cmd = ConfusionMatrixDisplay(matrix, display_labels=["Настоящая", "Подделанная"])
        plot_img_name = 'temp.png'
        cmd.plot().figure_.savefig(plot_img_name)
        self.label_matrix.setPixmap(QPixmap(plot_img_name))

        # tn, fp, fn, tp = matrix.ravel()
        # self.table_widget_matrix.setItem(0, 0, QTableWidgetItem(f"{tn}"))
        # self.table_widget_matrix.setItem(0, 1, QTableWidgetItem(f"{fp}"))
        # self.table_widget_matrix.setItem(1, 0, QTableWidgetItem(f"{fn}"))
        # self.table_widget_matrix.setItem(1, 1, QTableWidgetItem(f"{tp}"))

    def __train_test_model(self):
        self.plain_text_edit_log.clear()

        self.model = siamese_dist.SignatureNet()
        batches = self.spin_box_batch_size.value()
        avg_train_losses, avg_val_losses, std_train_losses, std_val_losses = self.model.fit(batches,
                                                                                            self.spin_box_epochs_count.value(),
                                                                                            self.plain_text_edit_log.appendPlainText)

        report, matrix = self.model.test(batches, self.plain_text_edit_log.appendPlainText)

        return avg_train_losses, avg_val_losses, std_train_losses, std_val_losses, report, matrix

    def __show_result(self, result):
        avg_train_losses, avg_val_losses, std_train_losses, std_val_losses, report, matrix = result

        self.__plot_graphs(avg_train_losses, avg_val_losses, std_train_losses, std_val_losses)
        self.__set_table_metrics(report)
        self.__show_confusion_matrix(matrix)

    def __thread_complete(self):
        self.plain_text_edit_log.appendPlainText(f"Конец обучения {datetime.now():%d.%m.%Y %H:%M:%S%z}")
        self.__set_enabled_train_controls(True)

    def start_training_task(self):
        self.__set_enabled_train_controls(False)

        worker = worker_calculate_data.Worker(self.__train_test_model)
        worker.signals.result.connect(self.__show_result)
        worker.signals.finished.connect(self.__thread_complete)

        # Execute
        self.threadpool.start(worker)
