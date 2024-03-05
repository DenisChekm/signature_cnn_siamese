import sys

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.uic import loadUi

from ui.pyqt6.train_window import TrainWindow
from ui.pyqt6.predict_window import PredictWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("./ui/pyqt6/main_view.ui", self)
        self.train_view = TrainWindow()
        self.stacked_widget.addWidget(self.train_view)
        self.train_view.push_button_show_predict_window.clicked.connect(self.go_to_predict)

        self.predict_view = PredictWindow()
        self.stacked_widget.addWidget(self.predict_view)
        self.predict_view.push_button_show_train_window.clicked.connect(self.go_to_train)

    def go_to_train(self):
        self.stacked_widget.setCurrentIndex(0)

    def go_to_predict(self):
        self.stacked_widget.setCurrentIndex(1)


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
