from time import sleep
from PyQt6.QtCore import QThread, pyqtSignal


class Worker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        for i in range(101):
            sleep(0.1)
            self.progress.emit(i)

        self.finished.emit()
