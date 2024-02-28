import sys

from PyQt6.QtWidgets import QApplication
from ui.pyqt6.train_window import TrainWindow


def main():
    app = QApplication([])

    window = TrainWindow()

    window.show()
    sys.exit(app.exec())
