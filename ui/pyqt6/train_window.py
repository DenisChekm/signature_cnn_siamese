from PyQt6.QtWidgets import QWidget
from PyQt6.uic import loadUi
from PyQt6.QtGui import QIcon

from model import siamese_bce


class TrainWindow(QWidget):
    def __init__(self):
        super(TrainWindow, self).__init__()
        loadUi("C:\\Users\\denle\\PycharmProjects\\signature_cnn_siamese\\ui\\pyqt6\\train_view.ui", self)

        self.setWindowTitle("Train")
        # self.setWindowIcon(QIcon("../../images/signature-icon-50.png"))

        self.push_button_train.clicked.connect(self.__train_model)

    # def __show_report_on_table(self, report):
    #     test_acc = report['accuracy']
    #     p = report['macro avg']['precision']
    #     r = report['macro avg']['recall']
    #     f1 = report['macro avg']['f1 - score']

    # def __show_matrix_on_table(self, matrix):
    #     pass

    def __train_model(self):
        std_losses, report, matrix = siamese_bce.fit(batch_size=self.spin_box_batch_size.value(),
                                     epochs_number=self.spin_box_epochs_count.value())
        last_loss = std_losses[-1]
        # self.entry_result.delete(0, tk.END)
        # self.entry_result.insert(0, str(last_loss))

        #__show_report_on_table(report)
        # __show_matrix_on_table(matrix)
