import sys

from PyQt6.QtWidgets import QApplication
from ui.pyqt6.train_window import TrainWindow
from ui.pyqt6.predict_window import PredictWindow


def main():
    app = QApplication([])

    # window = TrainWindow()
    window = PredictWindow()

    window.show()
    sys.exit(app.exec())
