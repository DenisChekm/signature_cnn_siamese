from datetime import datetime

from PyQt6.QtCore import QThreadPool
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget, QTableWidgetItem
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
        init_plot(self.widget_plot_acc, "График аккуратности", "Аккуратность")

    def __set_enabled_train_controls(self, isEnable: bool):
        self.push_button_train.setEnabled(isEnable)  # self.push_button_train.setText("Начать обучение")
        self.push_button_show_predict_window.setEnabled(isEnable)
        self.group_box_train_params.setEnabled(isEnable)

    def __plot_graphs(self, avg_train_losses, avg_val_losses, acc_train_list, acc_val_list):
        plot_graph(self.widget_plot_loss, avg_train_losses, avg_val_losses)
        plot_graph(self.widget_plot_acc, acc_train_list, acc_val_list)

    def __set_table_metrics(self, report: dict):
        acc = report['accuracy']
        label_real = report['0.0']
        label_forged = report['1.0']
        self.table_widget_report.setItem(0, 1, QTableWidgetItem(f"{acc:.4f}"))
        self.table_widget_report.setItem(0, 2, QTableWidgetItem(f"{label_real['precision']:.4f}"))
        self.table_widget_report.setItem(0, 3, QTableWidgetItem(f"{label_real['recall']:.4f}"))
        self.table_widget_report.setItem(0, 4, QTableWidgetItem(f"{label_real['f1-score']:.4f}"))

        self.table_widget_report.setItem(1, 1, QTableWidgetItem(f"{acc:.4f}"))
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
        avg_train_losses, avg_val_losses, acc_train_list, acc_val_list = self.model.fit(batches,
                                                                                        self.spin_box_epochs_count.value(),
                                                                                        self.plain_text_edit_log.appendPlainText)

        report, matrix = self.model.test(batches, self.plain_text_edit_log.appendPlainText)

        return avg_train_losses, avg_val_losses, acc_train_list, acc_val_list, report, matrix

    def __train_test_model_show_example(self):
        self.plain_text_edit_log.clear()

        ""
        avg_train_losses = [3.0898, 0.6251, 0.3902, 0.2718, 0.2054, 0.1719, 0.1440, 0.1160, 0.0996, 0.0792, 0.0612,
                            0.0527, 0.0446, 0.0388, 0.0304, 0.0261, 0.0220, 0.0228, 0.0198, 0.0183, 0.0177, 0.0161,
                            0.0167, 0.0140, 0.0140, 0.0124, 0.0120, 0.0118, 0.0106, 0.0104, 0.0093, 0.0090, 0.0089,
                            0.0079, 0.0083, 0.0080, 0.0078, 0.0074, 0.0073, 0.0071, 0.0062, 0.0060, 0.0060, 0.0061,
                            0.0058, 0.0056, 0.0063, 0.0050, 0.0046, 0.0051]
        acc_train_list = [0.5029, 0.5420, 0.5878, 0.6513, 0.7209, 0.7781, 0.8230, 0.8680, 0.8944, 0.9275, 0.9564,
                          0.9672, 0.9761, 0.9807, 0.9909, 0.9923, 0.9956, 0.9939, 0.9961, 0.9966, 0.9960, 0.9973,
                          0.9967, 0.9974, 0.9980, 0.9982, 0.9989, 0.9988, 0.9989, 0.9987, 0.9992, 0.9993, 0.9992,
                          0.9994, 0.9995, 0.9997, 0.9996, 0.9993, 0.9995, 0.9997, 0.9997, 0.9996, 0.9998, 0.9994,
                          0.9998, 0.9997, 0.9992, 0.9999, 1.0000, 0.9998]
        avg_val_losses = [0.7815, 0.7212, 0.6206, 0.4542, 0.3782, 0.2489, 0.2412, 0.2196, 0.1968, 0.2092, 0.1992,
                          0.1822, 0.1729, 0.1689, 0.1727, 0.1701, 0.1787, 0.1695, 0.1888, 0.1589, 0.1733, 0.1730,
                          0.1802, 0.1583, 0.1646, 0.1566, 0.1675, 0.1525, 0.1563, 0.1787, 0.1726, 0.1422, 0.1384,
                          0.1546, 0.1448, 0.1387, 0.1595, 0.1390, 0.1376, 0.1395, 0.1447, 0.1415, 0.1459, 0.1303,
                          0.1284, 0.1338, 0.1395, 0.1495, 0.1247, 0.1459]
        acc_val_list = [0.6257, 0.6335, 0.6335, 0.6636, 0.6716, 0.7041, 0.7199, 0.7119, 0.7263, 0.7140, 0.6977,
                        0.7296, 0.7178, 0.7469, 0.7488, 0.7379, 0.7330, 0.7405, 0.7244, 0.7720, 0.7670, 0.7450, 0.7384,
                        0.7725, 0.7450, 0.7550, 0.7417, 0.7834, 0.7931, 0.7498, 0.7379, 0.7876, 0.7992, 0.7732, 0.7862,
                        0.7988, 0.7656, 0.7933, 0.8113, 0.8040, 0.7936, 0.7995, 0.8071, 0.8243, 0.8338, 0.8196, 0.8000,
                        0.7829, 0.8404, 0.8099]

        # report = {'0.0': {'precision': 0.8512993262752647, 'recall': 0.8407794676806084, 'f1-score': 0.8460066953610712,
        #                   'support': 2104.0},
        #           '1.0': {'precision': 0.8438956197576887, 'recall': 0.8542452830188679, 'f1-score': 0.8490389123300516,
        #                   'support': 2120.0}, 'accuracy': 0.8475378787878788,
        #           'macro avg': {'precision': 0.8475974730164767, 'recall': 0.8475123753497382,
        #                         'f1-score': 0.8475228038455613, 'support': 4224.0},
        #           'weighted avg': {'precision': 0.8475834508450418, 'recall': 0.8475378787878788,
        #                            'f1-score': 0.8475285466807299, 'support': 4224.0}}
        # matrix = [[1769, 335],
        #           [309, 1811]]
        self.model = siamese_dist.SignatureNet()
        self.model.load_best_model()
        report, matrix = self.model.test(self.spin_box_batch_size.value(), self.plain_text_edit_log.appendPlainText)

        return avg_train_losses, avg_val_losses, acc_train_list, acc_val_list, report, matrix

    def __show_result(self, result):
        avg_train_losses, avg_val_losses, acc_train_list, acc_val_list, report, matrix = result

        self.__plot_graphs(avg_train_losses, avg_val_losses, acc_train_list, acc_val_list)
        self.__set_table_metrics(report)
        self.__show_confusion_matrix(matrix)

    def __thread_complete(self):
        self.plain_text_edit_log.appendPlainText(f"Конец обучения {datetime.now():%d.%m.%Y %H:%M:%S%z}")
        self.__set_enabled_train_controls(True)

    def start_training_task(self):
        self.__set_enabled_train_controls(False)

        worker = worker_calculate_data.Worker(self.__train_test_model)  # self.__train_test_model_show_example
        worker.signals.result.connect(self.__show_result)
        worker.signals.finished.connect(self.__thread_complete)

        # Execute
        self.threadpool.start(worker)
